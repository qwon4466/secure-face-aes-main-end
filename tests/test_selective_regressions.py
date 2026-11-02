import io
import secrets
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.selective_camera import MainCamera
from core.selective_crypto import GROUPS, RestoreError, protect, restore_frames, save_chunk
from core.selective_recognition import (CPUAdapter, Classifier, Policy,
                                        RegistrationFrameSelector,
                                        registration_face, stable_person_id,
                                        valid_embedding)


def _vector(value):
    stream = io.BytesIO()
    np.save(stream, np.full(512, value, dtype=np.float32), allow_pickle=False)
    return stream.getvalue()


def test_registration_samples_share_stable_person_id(work):
    assert stable_person_id('홍길동_연속_0') == '홍길동'
    assert stable_person_id('홍_길동_정면') == '홍_길동'
    assert stable_person_id('홍_길동_측면1(좌)') == '홍_길동'

    db = work / 'users.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE users (name TEXT, auth_group TEXT, vector BLOB)')
    for name, value in [('홍_길동_연속_0', 1), ('홍_길동_정면', 2)]:
        conn.execute('INSERT INTO users VALUES (?,?,?)', (name, 'staff', _vector(value)))
    conn.commit(); conn.close()
    adapter = CPUAdapter.__new__(CPUAdapter)
    adapter.reload_gallery(db)
    assert [person for person, _ in adapter.users] == ['홍_길동', '홍_길동']


def test_gallery_reload_sees_new_wal_commit_and_reports_valid_count(work):
    db=work/'wal-users.db';writer=sqlite3.connect(db)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('CREATE TABLE users (name TEXT, auth_group TEXT, vector BLOB)')
    writer.execute('INSERT INTO users VALUES (?,?,?)',('first','허가',_vector(1)))
    writer.commit()
    adapter=CPUAdapter.__new__(CPUAdapter);adapter.reload_gallery(db)
    assert len(adapter.users)==1
    writer.execute('INSERT INTO users VALUES (?,?,?)',('second','허가',_vector(2)))
    writer.commit();adapter.reload_gallery(db)
    assert len(adapter.users)==2
    assert adapter.gallery_info=={'total_rows':2,'valid_embeddings':2,
                                  'invalid_embeddings':0,'embedding_dimension':512}
    snapshot_users,snapshot_info=adapter._gallery_snapshot
    assert snapshot_users is adapter.users and snapshot_info is adapter.gallery_info
    writer.close()


def test_registration_quality_rejects_ambiguous_and_invalid_faces():
    frame=np.zeros((240,320,3),dtype=np.uint8)
    box=np.array([[80,40,240,220,.95]],dtype=np.float32)
    landmarks=np.array([[[120,90],[200,90],[160,130],[130,175],[190,175]]],dtype=np.float32)
    candidate,reason=registration_face(frame,box,landmarks)
    assert reason=='' and candidate['pose']=='front'
    assert valid_embedding(np.ones(512,dtype=np.float32),512)
    assert not valid_embedding(np.ones(256,dtype=np.float32),512)
    assert not valid_embedding(np.full(512,np.nan,dtype=np.float32),512)
    assert not valid_embedding(np.full(512,np.inf,dtype=np.float32),512)
    assert registration_face(frame,np.repeat(box,2,axis=0),
                             np.repeat(landmarks,2,axis=0))[1]=='multiple_faces'
    bad=landmarks.copy();bad[0,2,1]=70
    assert registration_face(frame,box,bad)[1]=='landmark_geometry_invalid'


def test_registration_selector_rejects_duplicate_frame_pose_and_generation():
    selector=RegistrationFrameSelector(pose_limit=2,pose_delta=.04)
    assert selector.check_frame(10,1)==''
    assert selector.check_frame(10,1)=='duplicate_frame'
    assert selector.accept_pose('front',0.0)
    assert not selector.accept_pose('front',0.02)
    assert selector.accept_pose('front',0.08)
    assert not selector.accept_pose('front',0.2)
    assert selector.check_frame(11,2)=='camera_generation_changed'


def test_tracking_survives_skipped_capture_ids_but_expires_by_time():
    classifier=Classifier(Policy(internal_streak=3,track_max_seconds=.8))
    def observation():
        return [{'bbox':[10,10,80,90],'quality':.99,'similarity':.9,
                 'identity':'registered','gallery_valid':True}]
    first=classifier.classify(observation(),10,observed_at=1.0)[0]
    second=classifier.classify(observation(),14,observed_at=1.2)[0]
    third=classifier.classify(observation(),21,observed_at=1.4)[0]
    assert first['track_id']==second['track_id']==third['track_id']
    assert third['group']=='internal' and third['streak']==3
    expired=classifier.classify(observation(),22,observed_at=2.3)[0]
    assert expired['track_id']!=third['track_id'] and expired['streak']==1


def test_authorization_hysteresis_and_final_storage_group():
    camera = MainCamera.__new__(MainCamera)
    camera.config = SimpleNamespace(
        FACE_AUTH_CONFIRM_MATCHES=2, FACE_AUTH_CONFIRM_WINDOW=3,
        FACE_AUTH_REVOKE_MISMATCHES=5, FACE_AUTH_HOLD_SECONDS=0.8,
        FACE_AUTH_RETAIN_THRESHOLD=0.40, MATCH_THRESHOLD=0.45,
    )
    states = {}
    def frame(identity, score, current):
        face = {'track_id': 1, 'bbox': [0, 0, 20, 20],
                'identity': identity, 'similarity': score,
                'gallery_valid': True, 'authorized': False, 'group': 'external'}
        current = camera._update_display_authorization([face], [face], current, 1.0)
        return face, current

    face, states = frame('홍_길동', 0.90, states)
    assert not face['authorized'] and face['group'] == 'external'
    face, states = frame('홍_길동', 0.46, states)
    assert face['authorized'] and face['group'] == 'internal'
    # One embedding failure is held, rather than becoming an outsider.
    face, states = frame(None, None, states)
    assert face['authorized'] and face['group'] == 'internal'
    for _ in range(5):
        face, states = frame('other-person', 0.10, states)
    assert not face['authorized'] and face['group'] == 'external'


def test_confirmed_authorization_is_not_transferred_to_new_identity():
    camera=MainCamera.__new__(MainCamera)
    camera.config=SimpleNamespace(
        FACE_AUTH_CONFIRM_MATCHES=2,FACE_AUTH_CONFIRM_WINDOW=3,
        FACE_AUTH_REVOKE_MISMATCHES=5,FACE_AUTH_HOLD_SECONDS=.8,
        FACE_AUTH_RETAIN_THRESHOLD=.40,MATCH_THRESHOLD=.45)
    states={}
    def apply(identity):
        observation={'bbox':[0,0,30,30],'identity':identity,'similarity':.9,
                     'gallery_valid':True,'quality':.99}
        face=dict(observation,track_id=1)
        next_states=camera._update_display_authorization([face],[observation],states,1.0)
        return face,next_states
    _,states=apply('person-a');authorized,states=apply('person-a')
    assert authorized['authorized']
    replacement,states=apply('person-b')
    assert not replacement['authorized']
    assert replacement['recognition_failure']=='confirmation_pending'


def test_detector_dropout_reuses_bbox_only_inside_hold_window():
    camera = MainCamera.__new__(MainCamera)
    camera.config = SimpleNamespace(FACE_AUTH_HOLD_SECONDS=0.8, FACE_AUTH_TRACK_IOU=0.35)
    previous = [{'bbox': [10, 10, 30, 30], 'quality': 0.95,
                'identity': '홍_길동', 'similarity': 0.9, 'gallery_valid': True}]
    held = camera._hold_detector_dropouts([], previous, 1.0, 1.5)
    assert len(held) == 1 and held[0]['detector_hold'] is True
    assert camera._hold_detector_dropouts([], previous, 1.0, 2.0) == []


def test_detector_hold_uses_per_face_real_timestamp_and_does_not_duplicate():
    camera = MainCamera.__new__(MainCamera)
    camera.config = SimpleNamespace(FACE_AUTH_HOLD_SECONDS=0.8, FACE_AUTH_TRACK_IOU=0.35)
    old = {'bbox': [10, 10, 50, 50], 'quality': 0.95,
           'identity': None, 'similarity': 0.1, 'gallery_valid': True,
           '_last_real_at': 1.0}
    held = camera._hold_detector_dropouts([], [old], 1.0, 1.5)
    assert held[0]['detector_hold'] and held[0]['_last_real_at'] == 1.0

    # A different face must not refresh this held bbox's lifetime.
    other = {'bbox': [200, 10, 240, 50], 'quality': 0.95,
             'identity': 'other', 'similarity': 0.9, 'gallery_valid': True}
    after_expiry = camera._hold_detector_dropouts([other], held, 2.0, 2.0)
    assert len(after_expiry) == 1 and after_expiry[0]['bbox'] == other['bbox']
    assert after_expiry[0]['detector_hold'] is False

    # A moved real detection supersedes the held copy, while two genuinely
    # current detections are both retained.
    moved = dict(other, bbox=[18, 12, 58, 52])
    two_people = camera._hold_detector_dropouts([moved, other], [old], 1.0, 1.4)
    assert len(two_people) == 2
    assert all(item['detector_hold'] is False for item in two_people)


def test_held_face_protects_frame_but_is_not_counted_as_a_new_person():
    camera = MainCamera.__new__(MainCamera)
    camera.lock = threading.RLock()
    camera.config = SimpleNamespace(CAMERA_STALE_SECONDS=3.0)
    camera.running = True
    camera.error = None
    camera.preview = b'jpeg'
    camera.last_preview_at = time.monotonic()
    camera.frame_sequence = 7
    camera.last_read_ms = 4.2
    camera.last_process_ms = 52.3
    camera.model_backend = 'test'
    camera.metrics = {}
    camera.faces = [
        {'authorized': True, 'detector_hold': False},
        {'authorized': False, 'detector_hold': True},
    ]

    stats = camera.get_stats()
    assert stats['employee_count'] == 1
    assert stats['unknown_count'] == 0
    assert stats['camera_ready'] and stats['camera_state'] == 'live'

    camera.last_preview_at -= 4.0
    stale = camera.get_stats()
    assert not stale['camera_ready'] and stale['camera_state'] == 'stale'

    debug = camera.get_debug_info()
    assert debug['face_diagnostics'][0]['detector_confidence'] is None
    assert debug['face_diagnostics'][1]['detector_hold'] is True
    assert debug['recognition_thresholds']['match'] == 0.45


def test_cpu_scrfd_threshold_validation_keeps_existing_safe_default():
    with pytest.raises(RestoreError,match='SCRFD'):
        CPUAdapter('', '', detector_threshold=0)


def test_operational_container_selective_restore_uses_groups(work):
    keys = {group: secrets.token_bytes(32) for group in GROUPS}
    originals = [np.full((48, 64, 3), 40 + i, dtype=np.uint8) for i in range(2)]
    faces = [[
        {'bbox': [2, 2, 20, 20], 'group': 'external', 'track_id': 1},
        {'bbox': [30, 2, 50, 20], 'group': 'internal', 'track_id': 2},
    ] for _ in originals]
    chunk = save_chunk(
        work, originals, faces,
        [{'capture_id': i, 'timestamp_ms': i * 100} for i in range(2)],
        keys, 'a' * 32, 10,
        final_path=work / '2026-01-01' / '00' / '00-00-00',
        duration=0.2,
    )
    assert {path.name for path in chunk.iterdir()} == {'encrypted_raw.bin.enc', 'encrypted_raw.bin.json'}
    protected, masks, _ = protect(originals[0], faces[0])
    for group, expected in [('external', 'external'), ('internal', 'internal')]:
        restored, _, groups = restore_frames(chunk, keys[group])
        assert groups == [expected]
        assert len(restored) == len(originals)
        allowed = masks[group]
        assert np.array_equal(restored[0][allowed], originals[0][allowed])
        assert np.array_equal(restored[0][~allowed], protected[~allowed])
