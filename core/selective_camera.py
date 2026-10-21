"""Camera interface for main.py: one raw read, selective recorder, protected monitor."""
import io
import os
from pathlib import Path
import sqlite3
import shutil
import threading
import time
import traceback
from collections import deque
import cv2
import numpy as np
from .selective_crypto import ROOT, RestoreError, generate_keys, keyset, private_dir, protect
from .selective_recognition import CPUAdapter, Classifier, Policy, SyntheticAdapter, synthetic_frame, iou, stable_person_id
from .selective_recording import Recorder
from .camera_manager import CameraManager, SELECTED
from .face_display import korean_font
from .detection_alerts import AlertTracker
from .access_policy import AccessEnv, evaluate, stored_floors

DB_PATH='security_system.db'
def _adapt_array(arr):
    buf=io.BytesIO();np.save(buf,arr,allow_pickle=False);return sqlite3.Binary(buf.getvalue())
def _convert_array(data):
    return np.load(io.BytesIO(data),allow_pickle=False)

class MainCamera:
    def __init__(self, config):
        self.config=config
        self.gallery_path=DB_PATH
        self.test=os.environ.get('SRX_TEST_MODE')=='synthetic'
        self.operational=not (self.test or os.environ.get('SRX_DATA_ROOT'))
        if self.operational:
            duration=getattr(config,'CHUNK_SECONDS',60)
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not 0<duration<=86400:
                raise RestoreError('CHUNK_SECONDS는 0보다 큰 유효한 초 값이어야 합니다')
        default_root=ROOT/'tests/runtime/main-server-test' if self.test else getattr(config,'RAW_DATA_DIR',ROOT/'raw_data')
        self.root=private_dir(Path(os.environ.get('SRX_DATA_ROOT',str(default_root))).absolute())
        self.keydir=self.root/('.keys' if self.operational else 'keys')
        if not self.keydir.exists():
            old=ROOT/'data/selective/main-server/keys'
            if self.operational and old.is_dir():
                private_dir(self.keydir)
                for name in ('internal.key','external.key','master.key'):
                    shutil.copyfile(old/name,self.keydir/name);os.chmod(self.keydir/name,0o600)
            else:generate_keys(self.keydir)
        self.keys=keyset(self.keydir)
        self.chunks=self.root if self.operational else private_dir(self.root/'chunks')
        self.lock=threading.RLock();self.thread=None;self.stop_event=threading.Event()
        self.preview=None;self.raw=None;self.detector=None;self.recognizer=None;self.adapter=None
        self.error=None;self.faces=[];self.metrics={};self.running=False;self.recorder=None
        # 비허가자 감지 이벤트: 모니터 팝업(alert_seq/recent_alerts)과 감지 이력(on_alert)에 사용
        self.alerts=self._alert_tracker();self.alert_seq=0;self.alert_events=deque(maxlen=20);self.on_alert=None
        # 시간·층별 출입 판정 (표시·알림 전용, 암호화 그룹과 무관)
        self.access_floors=list(getattr(config,'ACCESS_FLOORS',[1,2,3]))
        self.access_env=AccessEnv(self.root/'access_env.json',self.access_floors,
                                  camera_floor=getattr(config,'ACCESS_CAMERA_FLOOR',None),
                                  work_days=getattr(config,'WORK_DAYS',(0,1,2,3,4)),
                                  work_start=getattr(config,'WORK_START','09:00'),
                                  work_end=getattr(config,'WORK_END','18:00'))
        self.people={}
        self.model_backend='initializing'
        self.last_preview_at=None;self.frame_sequence=0
        self.last_read_ms=0.0;self.last_process_ms=0.0

    def _alert_tracker(self):
        return AlertTracker(getattr(self.config,'DETECT_ALERT_CONFIRM_SECONDS',1.0),
                            getattr(self.config,'DETECT_ALERT_REARM_SECONDS',5.0))

    def _record_alerts(self, alerts, faces, names):
        stamp=time.strftime('%Y-%m-%dT%H:%M:%S')
        in_view=sum(not f.get('allowed',False) for f in faces)
        with self.lock:
            for face in alerts:
                self.alert_seq+=1
                self.alert_events.append({'id':self.alert_seq,'time':stamp,'in_view':in_view,
                                          'reason':face.get('access_reason','')})
        if self.on_alert:
            for face in alerts:
                try:self.on_alert(face,names.get(face.get('track_id')))
                except Exception as exc:print('[Alert] 감지 이력 기록 실패:',exc,flush=True)

    def _load_people(self):
        """등록자별 그룹(가장 최근 등록 값)과 출입 허가 층. 층 정보가 없으면 전체 층 허가."""
        people={}
        try:
            conn=sqlite3.connect(Path(self.gallery_path).resolve().as_uri()+'?mode=ro',uri=True)
            try:
                for name,group in conn.execute('SELECT name, auth_group FROM users ORDER BY id'):
                    people[stable_person_id(name)]={'group':group,'floors':None}
                try:
                    for name,floors in conn.execute('SELECT name, floors FROM person_access'):
                        if name in people:people[name]['floors']=stored_floors(floors,self.access_floors)
                except sqlite3.OperationalError:
                    pass   # 층 정보 도입 전 DB
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            pass
        self.people=people

    def _apply_access(self, faces, display_states):
        """face['allowed'], face['access_reason'] 설정. 이름은 얼굴 기록(청크 메타데이터)에 넣지 않고 따로 돌려준다."""
        env=self.access_env.effective();people=self.people;names={}
        for face in faces:
            person=None
            if face.get('authorized',False):
                person=display_states.get(face.get('track_id'),{}).get('identity')
            face['allowed'],face['access_reason']=evaluate(
                None if person is None else people.get(person,{}),env)
            if person is not None:names[face.get('track_id')]=person
        return names

    def reload_db(self):
        if self.adapter is not None and not self.test:
            self.adapter.reload_gallery(self.gallery_path)
        self._load_people()

    def _adapter(self):
        if self.test:return SyntheticAdapter()
        modeldir=Path(getattr(self.config,'SELECTIVE_MODEL_DIR',Path.home()/'.insightface/models/buffalo_s'))
        detector=getattr(self.config,'SELECTIVE_DETECTOR_PATH','')
        recognizer=getattr(self.config,'SELECTIVE_RECOGNIZER_PATH','')
        if not detector or not recognizer:
            candidates=list(modeldir.glob('*.onnx'))
            detector=next((str(p) for p in candidates if 'scrfd' in p.name.lower() or p.name.lower().startswith('det_')),'')
            recognizer=next((str(p) for p in candidates if 'w600k' in p.name.lower() or 'rec' in p.name.lower()),'')
        detector_override=None
        if getattr(self.config,'USE_HAILO',False):
            try:
                from hailo_infer import HAILO_AVAILABLE, HailoSCRFD
                if not HAILO_AVAILABLE:
                    raise RuntimeError('hailo_platform을 현재 Python 환경에서 불러올 수 없습니다')
                detector_override=HailoSCRFD(
                    getattr(self.config,'SCRFD_HEF_PATH'),
                    conf_thresh=float(getattr(self.config,'HAILO_DET_THRESH',0.5)),
                )
                self.model_backend='Hailo-8L SCRFD + CPU ArcFace'
                print('[AI] Hailo-8L SCRFD 로드 성공',flush=True)
            except Exception as exc:
                self.model_backend='CPU SCRFD + CPU ArcFace (Hailo fallback)'
                print(f'[AI] Hailo SCRFD 초기화 실패: {exc} → CPU SCRFD 폴백',flush=True)
        else:
            self.model_backend='CPU SCRFD + CPU ArcFace'
        adapter=CPUAdapter(detector,recognizer,
                           self.gallery_path if Path(self.gallery_path).is_file() else None,
                           detector_override=detector_override,
                           detector_threshold=float(getattr(
                               self.config,'CPU_SCRFD_DET_THRESHOLD',0.6)))
        print(f'[AI] 모델 준비 완료: {self.model_backend}',flush=True)
        self.detector=adapter.detector;self.recognizer=adapter.recognizer
        return adapter

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():return
            self.stop_event.clear();self.running=True;self.error=None
            self.thread=threading.Thread(target=self._loop,name='selective-camera',daemon=True)
            self.thread.start()

    def _update_display_authorization(self, faces, observations, states, now):
        """Apply match confirmation + delayed revoke to prevent one-frame auth flicker."""
        confirm=max(1,int(getattr(self.config,'FACE_AUTH_CONFIRM_MATCHES',2)))
        window=max(confirm,int(getattr(self.config,'FACE_AUTH_CONFIRM_WINDOW',3)))
        revoke=max(1,int(getattr(self.config,'FACE_AUTH_REVOKE_MISMATCHES',5)))
        hold=max(0.0,float(getattr(self.config,'FACE_AUTH_HOLD_SECONDS',0.8)))
        threshold=float(getattr(self.config,'MATCH_THRESHOLD',0.45))
        retain=float(getattr(self.config,'FACE_AUTH_RETAIN_THRESHOLD',max(0.0,threshold-0.05)))
        next_states={}

        for face,observation in zip(faces,observations):
            track=face['track_id']
            score=face.get('similarity')
            identity=observation.get('identity')
            matched=(face.get('gallery_valid',False) and identity is not None and
                     score is not None and np.isfinite(score) and score>=threshold)

            previous=states.get(track,{
                'identity':None,'matches':0,'history':[],'misses':0,
                'authorized':False,'hold_until':0.0
            })
            state=dict(previous)

            if observation.get('detector_hold'):
                # Held detections keep the prior display/storage decision only.
                # They are not fresh ArcFace evidence and must not advance or
                # revoke authorization state.
                face['authorized']=bool(state.get('authorized',False))
                face['group']='internal' if face['authorized'] else 'external'
                next_states[track]=state
                continue

            valid=(face.get('gallery_valid',False) and identity is not None and
                   score is not None and np.isfinite(score))
            same_identity=valid and state.get('identity')==identity
            if matched:
                if not same_identity:
                    state['identity']=identity
                    state['history']=[]
                    state['matches']=0
                history=list(state.get('history',[]))[-window+1:]
                history.append(1)
                state['history']=history
                state['matches']=sum(history)
                state['misses']=0
                if state['matches']>=confirm:
                    state['authorized']=True
                    state['hold_until']=now+hold
            elif valid and same_identity and float(score)>=retain and state.get('authorized',False):
                # A small score dip from pose/landmark changes is retained.
                state['misses']=0
                state['hold_until']=now+hold
            elif valid:
                # Only a valid, sustained mismatch counts toward revocation.
                state['history']=[]
                state['matches']=0
                if state.get('authorized',False):
                    state['misses']=state.get('misses',0)+1
                    if state['misses']>=revoke:
                        state['authorized']=False
                        state['identity']=None
                        state['misses']=0
                else:
                    state['identity']=None
                    state['misses']=0
            elif not state.get('authorized',False):
                state['identity']=None
                state['history']=[]
                state['matches']=0
            # Missing/invalid embeddings are deliberately held for the short
            # continuity window and never counted as a mismatch.

            face['authorized']=bool(state.get('authorized',False))
            # Storage policy follows the final display decision.  There is no
            # operational third bucket that could make selective restore skip
            # an otherwise visible face.
            face['group']='internal' if face['authorized'] else 'external'
            next_states[track]=state

        return next_states

    def _hold_detector_dropouts(self, observations, previous_observations,
                                last_detected_at, now):
        """Reuse each bbox briefly, measured from its own last real detection."""
        hold=float(getattr(self.config,'FACE_AUTH_HOLD_SECONDS',0.8))
        threshold=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))
        current=[dict(item,detector_hold=False,_last_real_at=now) for item in observations]
        if not previous_observations or hold<=0:
            return current

        def same_region(a,b):
            if iou(a,b)>=threshold:
                return True
            aw,ah=a[2]-a[0],a[3]-a[1];bw,bh=b[2]-b[0],b[3]-b[1]
            if min(aw,ah,bw,bh)<=0:
                return False
            size_ratio=min(aw*ah,bw*bh)/max(aw*ah,bw*bh)
            dx=(a[0]+a[2]-b[0]-b[2])/2;dy=(a[1]+a[3]-b[1]-b[3])/2
            radius=0.45*max(aw,ah,bw,bh)
            return size_ratio>=0.4 and dx*dx+dy*dy<=radius*radius

        for old in previous_observations:
            last_real=float(old.get('_last_real_at',last_detected_at))
            if now-last_real>hold:
                continue
            # A nearby real detection supersedes the old bbox.  This only drops
            # the held copy; it never merges or removes current real detections.
            if any(same_region(old['bbox'],item['bbox']) for item in current):
                continue
            held=dict(old,quality=0.0,detector_hold=True,_last_real_at=last_real)
            current.append(held)
        return current

    def _loop(self):
        cap=None
        try:
            korean_font()
            self.adapter=self._adapter()
            self._load_people()
            classifier=Classifier(Policy(track_iou=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))))
            display_states={}
            previous_observations=[];last_detected_at=-1e9;last_ai_error_log=0.0
            fps=getattr(self.config,'SELECTIVE_FPS',10)
            duration=getattr(self.config,'CHUNK_SECONDS',None) if self.operational else None
            self.recorder=Recorder(self.chunks,self.keys,fps=fps,
                                   chunk_frames=getattr(self.config,'SELECTIVE_CHUNK_FRAMES',8),
                                   duration=duration,background_flush=True)
            source=getattr(self.config,'SELECTIVE_SOURCE','')
            if not source and getattr(self.config,'FORCE_VIDEO',False):source=getattr(self.config,'VIDEO_FALLBACK',None)
            is_file=bool(source)
            if not self.test:
                if not source:
                    cap=CameraManager(config=self.config,stop_event=self.stop_event)
                    while not self.stop_event.is_set():
                        try:
                            cap.open();self.error=None;break
                        except RuntimeError as exc:
                            self.error=str(exc);print('[Camera]',exc,flush=True)
                            self.stop_event.wait(3)
                else:
                    if not Path(source).is_file():raise RestoreError('설정한 입력 파일이 없습니다')
                    cap=cv2.VideoCapture(str(source))
                    if not cap.isOpened():raise RestoreError('파일 입력을 열 수 없습니다')
            capture=0;start=time.monotonic()
            limit=int(os.environ.get('SRX_TEST_FRAMES','24')) if self.test else None
            while not self.stop_event.is_set():
                tick=time.monotonic()
                if self.test:
                    if capture>=limit:break
                    frame=synthetic_frame(capture)
                else:
                    read_started=time.monotonic()
                    ok,frame=cap.read()
                    self.last_read_ms=(time.monotonic()-read_started)*1000
                    if not ok:
                        if is_file:break
                        if self.recorder.duration:self.recorder.abort()
                        else:self.recorder.flush()
                        classifier=Classifier(Policy(track_iou=float(getattr(self.config,'FACE_AUTH_TRACK_IOU',0.35))))
                        display_states={};previous_observations=[];last_detected_at=-1e9
                        self.alerts=self._alert_tracker()
                        with self.lock:
                            self.error='카메라 연결 끊김: 재연결 중';self.preview=None;self.raw=None;self.faces=[]
                        print('[Camera] 프레임 읽기 실패: 동일 장치 재연결 시도',flush=True)
                        while not self.stop_event.wait(2):
                            try:
                                cap.reconnect();self.error=None;break
                            except RuntimeError as exc:
                                self.error=str(exc);print('[Camera]',exc,flush=True)
                        continue
                width=getattr(self.config,'PROCESS_WIDTH',640)
                if width and frame.shape[1]>width:
                    height=int(frame.shape[0]*width/frame.shape[1])//2*2
                    frame=cv2.resize(frame,(width//2*2,height))
                try:
                    detected=self.adapter.observe(frame,capture)
                except Exception as exc:
                    with self.lock:
                        self.error=f'AI 프레임 처리 오류: {exc}';self.faces=[]
                    if tick-last_ai_error_log>=5:
                        print(f'[AI][ERROR] 프레임 {capture} 처리 실패: {exc}',flush=True)
                        last_ai_error_log=tick
                    self.stop_event.wait(.25)
                    continue
                if detected:
                    last_detected_at=tick
                observations=self._hold_detector_dropouts(
                    detected,previous_observations,last_detected_at,tick)
                faces=classifier.classify(observations,capture)
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) if cap is not None and is_file else (tick-start)*1000
                display_states=self._update_display_authorization(faces,observations,display_states,tick)
                names=self._apply_access(faces,display_states)
                alerts=self.alerts.update(faces,tick)
                if alerts:self._record_alerts(alerts,faces,names)
                previous_observations=list(observations)
                self.recorder.add(frame,faces,capture,timestamp)
                protected=protect(frame,faces)[0]
                ok,jpg=cv2.imencode('.jpg',protected)
                if not ok:raise RestoreError('보호 프레임 인코딩 실패')
                with self.lock:
                    self.preview=jpg.tobytes();self.raw=frame.copy();self.faces=faces
                    self.last_preview_at=time.monotonic();self.frame_sequence+=1
                    self.last_process_ms=(self.last_preview_at-tick)*1000
                    self.error=None;self.metrics=self.recorder.stats()
                capture+=1
                self.stop_event.wait(max(0,1/fps-(time.monotonic()-tick)))
            report=self.recorder.finish()
            with self.lock:self.metrics=report
        except Exception as exc:
            if self.recorder:self.recorder.close()
            with self.lock:self.error=str(exc)
            print(f'[Camera][ERROR] 처리 스레드 종료: {exc}',flush=True)
            traceback.print_exc()
        finally:
            if cap:cap.release()
            with self.lock:self.running=False

    def stop(self):
        self.stop_event.set()
        if self.thread:
            # The systemd unit allows 30 seconds for shutdown.  Give an active
            # background chunk save time to finish before native model threads
            # are torn down with the Python process.
            self.thread.join(timeout=25)
            if self.thread.is_alive():
                print('[Camera][WARN] 종료 대기 초과: 녹화 청크 저장/카메라 읽기 상태 확인 필요',flush=True)

    def get_jpeg(self):
        with self.lock:return self.preview
    def get_raw_jpeg(self):
        with self.lock:
            if self.raw is None:return None
            return cv2.imencode('.jpg',self.raw)[1].tobytes()
    def capture_raw_frame(self):
        with self.lock:return None if self.raw is None else self.raw.copy()
    def get_stats(self):
        with self.lock:
            age=None if self.last_preview_at is None else max(0.0,time.monotonic()-self.last_preview_at)
            stale=float(getattr(self.config,'CAMERA_STALE_SECONDS',3.0))
            ready=(self.running and self.error is None and self.preview is not None and
                   age is not None and age<=stale)
            state=('live' if ready else 'error' if self.error else
                   'stale' if self.running and self.preview is not None else
                   'starting' if self.running else 'stopped')
            real_faces=[f for f in self.faces if not f.get('detector_hold',False)]
            return {'employee_count':sum(f.get('authorized',False) for f in real_faces),
                    'unknown_count':sum(not f.get('authorized',False) for f in real_faces),
                    'denied_count':sum(not f.get('allowed',False) for f in real_faces),
                    'detected_face_count':len(real_faces),
                    'held_face_count':sum(f.get('detector_hold',False) for f in self.faces),
                    'recording':ready,'error':self.error,'camera_ready':ready,
                    'camera_state':state,'model_backend':self.model_backend,
                    'frame_sequence':self.frame_sequence,
                    'alert_seq':self.alert_seq,'recent_alerts':list(self.alert_events),
                    'access_env':self.access_env.get(),
                    'frame_age_ms':None if age is None else round(age*1000,1),
                    'camera_read_ms':round(self.last_read_ms,1),
                    'frame_process_ms':round(self.last_process_ms,1),
                    'selective_restore':True,'camera':dict(SELECTED),**self.metrics}
    def get_save_progress(self):
        with self.lock:return self.recorder.progress() if self.recorder else {'in_progress':False}
    def get_perf(self):
        with self.lock:return self.metrics.copy()
    def get_debug_info(self):
        stats=self.get_stats()
        with self.lock:
            stats['face_diagnostics']=[{
                'track_id':f.get('track_id'),
                'bbox':f.get('bbox'),
                'detector_confidence':f.get('quality'),
                'arcface_similarity':f.get('similarity'),
                'recognition_status':f.get('recognition_status','unknown'),
                'authorized':bool(f.get('authorized',False)),
                'allowed':bool(f.get('allowed',False)),
                'access_reason':f.get('access_reason',''),
                'detector_hold':bool(f.get('detector_hold',False)),
            } for f in self.faces]
        stats['recognition_thresholds']={
            'match':float(getattr(self.config,'MATCH_THRESHOLD',0.45)),
            'retain':float(getattr(self.config,'FACE_AUTH_RETAIN_THRESHOLD',0.40)),
            'hailo_scrfd':float(getattr(self.config,'HAILO_DET_THRESH',0.5)),
            'cpu_scrfd':float(getattr(self.config,'CPU_SCRFD_DET_THRESHOLD',0.6)),
        }
        return stats
    def web_state(self):
        stats=self.get_stats()
        with self.lock:return {'recording':stats['recording'],'error':self.error,
                               'faces':self.faces.copy(),'stats':self.metrics.copy()}

    def list_assets(self):
        from .selective_index import completed, summary
        result=[]
        for path,meta in completed(self.chunks):
            item=summary(path,meta,self.chunks)
            result.append({'chunk_id':item['id'],'display_name':item['name'],'date':item['date'],'start_time':item['started_at'][11:19],
                'duration_seconds':item['duration_seconds'],'frame_count':item['frames'],
                'total_faces':sum(item['groups'].values()),'complete':True,'format':'selective',
                'fps':item['fps'],'resolution':item['shape'][1::-1],'path':item['path']})
        return result
