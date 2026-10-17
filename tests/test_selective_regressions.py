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
from core.selective_recognition import CPUAdapter, stable_person_id


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
