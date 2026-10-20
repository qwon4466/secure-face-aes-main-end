from pathlib import Path

from core.detection_alerts import AlertTracker


def face(track, bbox=(0, 0, 50, 50), authorized=False, hold=False):
    return {'track_id': track, 'bbox': list(bbox), 'authorized': authorized, 'detector_hold': hold}


def run(tracker, frames, step=0.1):
    """frames: 프레임별 얼굴 목록. 프레임마다 확정된 track_id 목록을 돌려준다."""
    return [[f['track_id'] for f in tracker.update(faces, i * step)] for i, faces in enumerate(frames)]


def test_unknown_face_alerts_once_after_confirm_time():
    out = run(AlertTracker(1.0, 5.0), [[face(1)]] * 30)
    assert sum(out, []) == [1]
    assert out.index([1]) == 10   # 1.0초 지속 후 확정


def test_registered_person_recognized_late_is_not_alerted():
    # 처음 3프레임(0.3초)은 인식 확정 전이라 비허가로 보이다가 허가로 바뀐다.
    frames = [[face(1)]] * 3 + [[face(1, authorized=True)]] * 30
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == []


def test_brief_flicker_resets_and_does_not_alert():
    frames = ([[face(1)]] * 5 + [[]]) * 5
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == []


def test_track_change_in_same_place_is_not_realerted():
    frames = [[face(1)]] * 15 + [[face(2, (5, 5, 55, 55))]] * 15
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == [1]


def test_new_person_elsewhere_alerts_separately():
    frames = [[face(1)]] * 15 + [[face(1), face(2, (200, 0, 250, 50))]] * 15
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == [1, 2]


def test_returning_after_rearm_time_alerts_again():
    frames = [[face(1)]] * 15 + [[]] * 60 + [[face(2)]] * 15
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == [1, 2]


def test_detector_hold_does_not_confirm_alone():
    frames = [[face(1)]] + [[face(1, hold=True)]] * 20
    assert sum(run(AlertTracker(1.0, 5.0), frames), []) == []


def test_monitor_alert_popup_closes_only_with_confirm():
    html = (Path(__file__).parents[1] / 'templates' / 'monitor.html').read_text(encoding='utf-8')
    assert 'id="alertOverlay"' in html and 'id="alertConfirm"' in html
    assert "alertOverlay.hidden = true" in html
    assert html.count("alertOverlay.hidden = true") == 1   # 확인 버튼 외에 닫는 경로 없음
    assert 'setTimeout(() => el.remove()' not in html      # 자동으로 사라지는 토스트 제거
    assert 'alert_seq' in html
