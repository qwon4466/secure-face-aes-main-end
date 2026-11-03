import io
import queue
import sqlite3
import threading
from types import SimpleNamespace
import time
import numpy as np
import pytest
from core.selective_recognition import CPUAdapter
from core.selective_camera import MainCamera
from core.selective_crypto import inspect_chunk,restore_frames,RestoreError


def test_live_frame_buffer_is_bounded_and_keeps_latest():
    camera=MainCamera.__new__(MainCamera)
    camera.lock=threading.RLock();camera.frame_queue=queue.Queue(maxsize=2)
    camera.camera_buffer_drops=0
    camera._put_latest_frame(('first',))
    camera._put_latest_frame(('second',))
    camera._put_latest_frame(('latest',))
    assert camera.frame_queue.qsize()==2
    assert camera.frame_queue.get_nowait()==('second',)
    assert camera.frame_queue.get_nowait()==('latest',)
    assert camera.camera_buffer_drops==1

def test_main_camera_records_and_restarts(work,monkeypatch):
    monkeypatch.setenv('SRX_TEST_MODE','synthetic');monkeypatch.setenv('SRX_TEST_FRAMES','9');monkeypatch.setenv('SRX_DATA_ROOT',str(work))
    c=MainCamera(SimpleNamespace(SELECTIVE_FPS=120,SELECTIVE_CHUNK_FRAMES=8,PROCESS_WIDTH=640))
    for run in range(2):
        c.start();c.thread.join(timeout=10)
        assert not c.thread.is_alive() and c.error is None
        assert c.get_jpeg() is not None and c.capture_raw_frame().shape==(240,320,3)
        assert not hasattr(c,'_raw_recorder')
    folders=[p for p in c.chunks.iterdir() if p.is_dir()]
    assert len(folders)==4
    for p in folders:inspect_chunk(p)
    c.stop()

def test_main_model_failure_visible(work,monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE',raising=False);monkeypatch.setenv('SRX_DATA_ROOT',str(work))
    c=MainCamera(SimpleNamespace(SELECTIVE_MODEL_DIR=str(work/'no-models')))
    c.start();c.thread.join(timeout=5)
    assert not c.running and c.error and not list(c.chunks.iterdir())
    assert c.web_state()['error'] and not c.get_stats()['recording']

def test_gallery_reload(work):
    db=work/'users.db';conn=sqlite3.connect(db)
    conn.execute('CREATE TABLE users (name TEXT, auth_group TEXT, vector BLOB)')
    data=io.BytesIO();np.save(data,np.ones(512,dtype=np.float32),allow_pickle=False)
    conn.execute('INSERT INTO users VALUES (?,?,?)',('test','staff',data.getvalue()));conn.commit()
    a=CPUAdapter.__new__(CPUAdapter);a.reload_gallery(db);assert len(a.users)==1
    conn.execute('DELETE FROM users');conn.commit();a.reload_gallery(db);assert a.users==[]
    conn.close()

def test_invalid_operational_chunk_seconds_fails_before_start(monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE',raising=False);monkeypatch.delenv('SRX_DATA_ROOT',raising=False);monkeypatch.delenv('SRX_VIDEO_SOURCE',raising=False)
    with pytest.raises(RestoreError,match='CHUNK_SECONDS'):
        MainCamera(SimpleNamespace(CHUNK_SECONDS=0))


def test_main_camera_recovers_after_capture_read_failure(work, monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE', raising=False)
    monkeypatch.setenv('SRX_DATA_ROOT', str(work))
    frame = np.random.default_rng(7).integers(0, 255, (24, 32, 3), dtype=np.uint8)

    class Adapter:
        detector = object()
        recognizer = object()
        def observe(self, image, capture_id):
            return []

    class Capture:
        def __init__(self, config=None, stop_event=None):
            self.stop_event = stop_event
            self.reads = 0
            self.reconnects = 0
        def open(self):
            return self
        def read(self):
            self.reads += 1
            time.sleep(.01)
            if self.reads == 3:
                return False, None
            if self.reads >= 12:
                self.stop_event.set()
            return True, frame.copy()
        def reconnect(self):
            self.reconnects += 1
            return self
        def get(self, prop):
            return 0
        def release(self):
            pass

    capture = Capture()
    def camera_manager(**kwargs):
        capture.stop_event = kwargs.get('stop_event')
        return capture
    monkeypatch.setattr('core.selective_camera.CameraManager', camera_manager)
    monkeypatch.setattr(MainCamera, '_adapter', lambda self: Adapter())
    camera = MainCamera(SimpleNamespace(
        SELECTIVE_FPS=120, SELECTIVE_CHUNK_FRAMES=8, PROCESS_WIDTH=640,
        FACE_AUTH_TRACK_IOU=0.35, FACE_AUTH_HOLD_SECONDS=0.8,
        CAMERA_STALE_SECONDS=3.0,
    ))
    camera.start()
    camera.thread.join(timeout=8)

    assert not camera.thread.is_alive()
    # A single transient read failure must not tear down the camera.
    assert capture.reconnects == 0
    assert camera.frame_sequence >= 2
    assert camera.get_jpeg() is not None
    assert camera.error is None


def test_main_camera_reconnects_only_after_sustained_failures(work, monkeypatch):
    monkeypatch.delenv('SRX_TEST_MODE', raising=False)
    monkeypatch.setenv('SRX_DATA_ROOT', str(work))
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class Adapter:
        detector = object(); recognizer = object()
        def observe(self, image, capture_id): return []

    class Capture:
        generation = 1
        def __init__(self, config=None, stop_event=None):
            self.stop_event=stop_event;self.reads=0;self.reconnects=0
        def open(self): return self
        def read(self):
            self.reads+=1;time.sleep(.01)
            if 3<=self.reads<=8:return False,None
            if self.reads>=16:self.stop_event.set()
            return True,frame.copy()
        def reconnect(self):
            self.reconnects+=1;self.generation+=1;self.reads=9;return self
        def release(self): pass

    capture=Capture()
    def camera_manager(**kwargs):
        capture.stop_event=kwargs.get('stop_event');return capture
    monkeypatch.setattr('core.selective_camera.CameraManager',camera_manager)
    monkeypatch.setattr(MainCamera,'_adapter',lambda self:Adapter())
    camera=MainCamera(SimpleNamespace(
        SELECTIVE_FPS=120,SELECTIVE_CHUNK_FRAMES=8,PROCESS_WIDTH=640,
        FACE_AUTH_TRACK_IOU=0.35,FACE_AUTH_HOLD_SECONDS=0.8,
        CAMERA_STALE_SECONDS=3.0,CAMERA_RECONNECT_FAILURES=3,
        CAMERA_RECONNECT_GRACE_SECONDS=0,CAMERA_READ_RETRY_SECONDS=.01,
        CAMERA_RECONNECT_BACKOFF_INITIAL=.01,CAMERA_RECONNECT_BACKOFF_MAX=.02,
    ))
    camera.start();camera.thread.join(timeout=8)
    assert not camera.thread.is_alive()
    assert capture.reconnects==1
    assert camera.camera_reconnect_successes==1
    assert camera.camera_generation==2
