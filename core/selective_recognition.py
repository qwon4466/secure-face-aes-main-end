"""Read-only legacy adapter. No import of config/main/camera_stream modules."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import re
import numpy as np
from .selective_crypto import ROOT, RestoreError

PROJECT=ROOT

# Approved AST digest of camera_stream._cosine_sim.  Keep this pin in sync only
# after reviewing the function's numerical behaviour; a source edit must not
# silently become trusted.
_COSINE_AST_SHA256 = '21b3ad196d99c6547c36ddb24136bffdb9cfeca8cab8cd60022eb5be97058b4e'

# Registration appends only these known suffixes.  Keep user supplied
# underscores intact (for example ``unit_a`` remains ``unit_a``).
_PERSON_SUFFIXES = (
    re.compile(r'_연속_\d+$'),
    re.compile(r'_(?:정면|좌측면|우측면)$'),
    re.compile(r'_측면[12]\((?:좌|우)\)$'),
)

def stable_person_id(name):
    value = str(name).strip()
    for suffix in _PERSON_SUFFIXES:
        value = suffix.sub('', value)
    return value or str(name)

def legacy_components():
    """Use the original interface and unchanged cosine function, after hash checks.

    Compile only the audited pure function, without importing camera hardware modules.
    Its AST is pinned separately so camera initialization can evolve independently.
    """
    baseline=json.loads((ROOT/'docs/selective-original-baseline.json').read_text())['files']
    for name in ('detection/detector_base.py',):
        if hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()!=baseline[name]:
            raise RestoreError('기존 어댑터 소스가 변경되었습니다; 재검토가 필요합니다')
    name='srx._legacy_detector_base'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PROJECT/'detection/detector_base.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    tree=ast.parse((PROJECT/'camera_stream.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cosine_sim')
    if hashlib.sha256(ast.dump(node).encode()).hexdigest() != _COSINE_AST_SHA256:
        raise RestoreError('cosine 함수 변경: 재검토 필요')
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PROJECT/'camera_stream.py'),'exec'),scope)
    return sys.modules[name].DetectionResult,scope['_cosine_sim']

@dataclass(frozen=True)
class Policy:
    internal_score: float=0.55
    external_score: float=0.25
    quality: float=0.8
    internal_streak: int=3
    external_streak: int=5
    track_iou: float=0.35
    track_max_seconds: float=0.8

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return x*y/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x*y)

def valid_embedding(value, expected_dimension=None):
    """Validate ArcFace output without changing its feature space."""
    if value is None:return False
    vector=np.asarray(value)
    return (vector.dtype.kind=='f' and vector.size in (128,256,512) and
            (expected_dimension is None or vector.size==expected_dimension) and
            np.isfinite(vector).all() and np.linalg.norm(vector)>1e-8)

def registration_face(frame, boxes, landmarks, min_confidence=0.6, min_face_size=64):
    """Validate one unambiguous SCRFD face and return bbox, landmarks and pose."""
    if boxes is None or len(boxes)==0:return None,'face_not_detected'
    if len(boxes)!=1:return None,'multiple_faces'
    if landmarks is None or len(landmarks)!=1:return None,'landmarks_missing'
    box=np.asarray(boxes[0]);lm=np.asarray(landmarks[0],dtype=np.float32)
    if box.size<5 or not np.isfinite(box[:5]).all() or float(box[4])<min_confidence:
        return None,'low_detection_confidence'
    h,w=frame.shape[:2]
    x1,y1,x2,y2=[float(v) for v in box[:4]]
    bw,bh=x2-x1,y2-y1
    if bw<min_face_size or bh<min_face_size:return None,'face_too_small'
    if lm.shape!=(5,2) or not np.isfinite(lm).all():return None,'landmarks_invalid'
    margin=.15*max(bw,bh)
    if (np.any(lm[:,0]<max(0,x1-margin)) or np.any(lm[:,0]>min(w-1,x2+margin)) or
            np.any(lm[:,1]<max(0,y1-margin)) or np.any(lm[:,1]>min(h-1,y2+margin))):
        return None,'landmarks_outside_face'
    eye_mid=(lm[0]+lm[1])/2;mouth_mid=(lm[3]+lm[4])/2
    eye_distance=float(np.linalg.norm(lm[0]-lm[1]))
    mouth_distance=float(np.linalg.norm(lm[3]-lm[4]))
    if (eye_distance<max(8.0,bw*.15) or mouth_distance<max(5.0,bw*.08) or
            not eye_mid[1]<lm[2,1]<mouth_mid[1]):
        return None,'landmark_geometry_invalid'
    yaw=float((lm[2,0]-eye_mid[0])/max(eye_distance,1e-6))
    pose='front' if abs(yaw)<=.12 else ('left' if yaw<0 else 'right')
    return {'bbox':[int(x1),int(y1),int(x2),int(y2)],'landmarks':lm,
            'confidence':float(box[4]),'pose':pose,'yaw':yaw},''

class RegistrationFrameSelector:
    """Track unique camera frames and diverse landmark poses for one capture."""
    def __init__(self,pose_limit=8,pose_delta=.04):
        self.pose_limit=max(1,int(pose_limit));self.pose_delta=max(0.0,float(pose_delta))
        self.last_sequence=None;self.generation=None
        self.pose_yaws={'front':[],'left':[],'right':[]}

    def check_frame(self,sequence,generation):
        if sequence==self.last_sequence:return 'duplicate_frame'
        self.last_sequence=sequence
        if self.generation is None:self.generation=generation
        elif generation!=self.generation:return 'camera_generation_changed'
        return ''

    def accept_pose(self,pose,yaw):
        values=self.pose_yaws[pose]
        if len(values)>=self.pose_limit or any(abs(yaw-old)<self.pose_delta for old in values):
            return False
        values.append(yaw);return True

class Classifier:
    def __init__(self, policy=Policy()):
        self.policy=policy; self.previous=[]; self.next_id=0
        self.last_capture=None;self.last_observed_at=None

    def classify(self, observations, capture_id, observed_at=None):
        p=self.policy
        if observed_at is None:
            continuous=(self.last_capture is not None and capture_id==self.last_capture+1)
        else:
            continuous=(self.last_capture is not None and capture_id>self.last_capture and
                        self.last_observed_at is not None and observed_at>=self.last_observed_at and
                        observed_at-self.last_observed_at<=p.track_max_seconds)
        previous=self.previous if continuous else []
        matches=[[j for j,old in enumerate(previous) if iou(o['bbox'],old['bbox'])>=p.track_iou] for o in observations]
        usage={j:sum(j in row for row in matches) for j in range(len(previous))}
        current=[]
        for o,candidates in zip(observations,matches):
            old=previous[candidates[0]] if len(candidates)==1 and usage[candidates[0]]==1 else None
            score=o.get('similarity'); identity=o.get('identity')
            reliable=o.get('quality',0)>=p.quality and o.get('gallery_valid',False) and score is not None and np.isfinite(score)
            if o.get('detector_hold') and old is not None:
                # A held bbox is protection continuity, not new recognition
                # evidence.  Preserve the track without extending its lifetime.
                evidence=old['evidence'];token=old['token'];streak=old['streak']
            else:
                evidence='master'
                if reliable and score>=p.internal_score and identity:
                    evidence='internal'
                elif reliable and score<=p.external_score:
                    evidence='external'
                token=identity if evidence=='internal' else evidence
                consistent=old is not None and old['evidence']==evidence and old['token']==token
                streak=old['streak']+1 if consistent else 1
            if old is None:
                self.next_id+=1
            track=old['track_id'] if old else self.next_id
            needed=p.internal_streak if evidence=='internal' else p.external_streak
            group=evidence if evidence!='master' and streak>=needed else 'master'
            current.append({**o,'group':group,'track_id':track,'streak':streak,'evidence':evidence,'token':token,
                            'reason':'consistent' if group!='master' else 'uncertain_or_track_reset'})
        self.previous=current;self.last_capture=capture_id;self.last_observed_at=observed_at
        return [{k:v for k,v in o.items() if k not in ('identity','token','_last_real_at')} for o in current]

class CPUAdapter:
    """Same SCRFD/ArcFace detector.detect + norm_crop + get_feat as CameraProcessor._process.
    Explicit local ONNX models only; no FaceAnalysis model downloader or legacy recorder.
    """
    def __init__(self, detector_path, recognizer_path, gallery=None,
                 detector_override=None, detector_threshold=0.6):
        self.Result,self.cosine=legacy_components()
        if not isinstance(detector_threshold,(int,float)) or not 0<detector_threshold<=1:
            raise RestoreError('SCRFD 탐지 임계값은 0과 1 사이여야 합니다')
        paths=[Path(recognizer_path)]
        if detector_override is None:
            paths.append(Path(detector_path))
        if any(not p.is_file() or p.suffix!='.onnx' for p in paths):
            raise RestoreError('실제 SCRFD / ArcFace ONNX 모델 파일을 지정하세요')
        from insightface.model_zoo import get_model
        from insightface.utils import face_align
        self.align=face_align.norm_crop
        self.detector=(detector_override if detector_override is not None else
                       get_model(str(detector_path),providers=['CPUExecutionProvider']))
        self.recognizer=get_model(str(recognizer_path),providers=['CPUExecutionProvider'])
        if not hasattr(self.detector,'detect') or not hasattr(self.recognizer,'get_feat'):
            raise RestoreError('SCRFD 탐지 모델 / ArcFace 인식 모델 종류를 확인하세요')
        if detector_override is None:
            self.detector.prepare(ctx_id=-1,input_size=(640,640),det_thresh=float(detector_threshold))
        self.recognizer.prepare(ctx_id=-1)
        self.reload_gallery(gallery)

    def reload_gallery(self, gallery):
        users=[];candidates=[];total=0;invalid=0;dimension=None
        if gallery:
            gp=Path(gallery).resolve(strict=True)
            # ``immutable=1`` may ignore newly committed WAL content.  A normal
            # read-only connection sees the registration transaction that just
            # committed without granting this path write access.
            conn=sqlite3.connect(gp.as_uri()+'?mode=ro',uri=True,timeout=5)
            try:
                conn.execute('PRAGMA query_only=ON')
                for name,group,blob in conn.execute('SELECT name, auth_group, vector FROM users LIMIT 10001'):
                    total+=1
                    if total>10000:
                        raise RestoreError('등록 DB 크기 / 벡터 형식 오류')
                    if not isinstance(blob,bytes) or len(blob)>16384:
                        invalid+=1;continue
                    try:vec=np.load(io.BytesIO(blob),allow_pickle=False)
                    except Exception:invalid+=1;continue
                    valid=valid_embedding(vec)
                    if not valid:invalid+=1;continue
                    candidates.append((stable_person_id(name),vec))
            finally:
                conn.close()
        if candidates:
            counts={size:sum(vec.size==size for _,vec in candidates)
                    for size in {vec.size for _,vec in candidates}}
            dimension=int(max(counts,key=lambda size:(counts[size],size)))
            users=[item for item in candidates if item[1].size==dimension]
            invalid+=len(candidates)-len(users)
        # The recognition thread takes this pair through one local reference,
        # so vectors and their dimension metadata always come from one reload.
        info={'total_rows':total,'valid_embeddings':len(users),
              'invalid_embeddings':invalid,'embedding_dimension':dimension}
        self._gallery_snapshot=(users,info)
        # Keep the public attributes for existing callers and admin diagnostics.
        self.users=users;self.gallery_info=info

    def observe(self,frame,capture_id):
        boxes,landmarks=self.detector.detect(frame,max_num=0,metric='default')
        if boxes is None:
            return []
        out=[]; h,w=frame.shape[:2]
        users,gallery_info=self._gallery_snapshot
        gallery_dimension=gallery_info.get('embedding_dimension')
        for i,b in enumerate(boxes):
            coords=[max(0,int(b[0])),max(0,int(b[1])),min(w,int(b[2])),min(h,int(b[3]))]
            if coords[0]>=coords[2] or coords[1]>=coords[3]:
                raise RestoreError('탐지 bbox 오류')
            lm=landmarks[i] if landmarks is not None else None
            det=self.Result(coords,float(b[4]),lm)
            score=None; identity=None
            recognition_status=('no_registered_users' if not users else
                                'landmarks_missing')
            gallery_valid=bool(users)
            if lm is not None and np.isfinite(lm).all() and users:
                try:
                    emb=np.asarray(self.recognizer.get_feat(
                        self.align(frame,landmark=lm,image_size=112)))
                    if valid_embedding(emb,gallery_dimension):
                        ranked=[(self.cosine(emb,v),n) for n,v in users]
                        score,identity=max(ranked)
                        recognition_status='compared'
                    elif emb.size!=gallery_dimension:
                        gallery_valid=False
                        recognition_status='embedding_dimension_mismatch'
                    else:
                        recognition_status='embedding_failed'
                except Exception as exc:
                    recognition_status='embedding_failed'
                    print('[ArcFace] 임베딩 생성 실패: 비허가자 표시 / 전체키 전용:',type(exc).__name__,flush=True)
            out.append({'bbox':det.bbox,'quality':det.conf,'similarity':score,'identity':identity,
                        'gallery_valid':gallery_valid,'recognition_status':recognition_status})
        return out

class SyntheticAdapter:
    """ONLY deterministic test scenes, never an actual face detector."""
    def observe(self,frame,capture_id):
        if frame.shape!=(240,320,3):
            raise RestoreError('합성 입력은 320×240이어야 합니다')
        return [
            {'bbox':[20,30,60,90],'quality':0.99,'similarity':0.1,'identity':None,'gallery_valid':True},
            {'bbox':[110,30,150,90],'quality':0.99,'similarity':0.9,'identity':'test-employee','gallery_valid':True},
            {'bbox':[230,30,270,90],'quality':0.5,'similarity':None,'identity':None,'gallery_valid':True},
            {'bbox':[80,150,130,200],'quality':0.99,'similarity':0.1,'identity':None,'gallery_valid':True},
            {'bbox':[120,150,165,200],'quality':0.99,'similarity':0.9,'identity':'test-employee-2','gallery_valid':True}]

def synthetic_frame(i):
    rng=np.random.default_rng(1729+i)
    return rng.integers(0,256,size=(240,320,3),dtype=np.uint8)
