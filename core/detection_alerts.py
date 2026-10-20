"""비허가자 감지 이벤트 판정.

모니터 팝업과 감지 이력(/audit)이 같은 기준을 쓰도록 서버에서 한 번만 판정한다.

- 등록자도 처음 몇 프레임은 인식 확정 전이라 비허가로 보인다. 그래서 같은
  추적 번호가 confirm_seconds 동안 계속 비허가일 때만 이벤트로 확정한다.
- 한 사람이 화면에 머무는 동안은 한 번만 알린다. 빠르게 움직여 추적 번호가
  바뀌어도, 최근 rearm_seconds 안에 알린 얼굴 영역과 겹치면 같은 사람으로 본다.
"""


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class AlertTracker:
    def __init__(self, confirm_seconds=1.0, rearm_seconds=5.0):
        self.confirm = max(0.0, float(confirm_seconds))
        self.rearm = max(0.0, float(rearm_seconds))
        self.pending = {}   # track_id -> 처음 비허가로 보인 시각
        self.alerted = {}   # track_id -> (마지막 bbox, 마지막으로 본 시각)

    def update(self, faces, now):
        """현재 프레임 얼굴 목록을 반영하고, 새로 확정된 비허가 얼굴 목록을 돌려준다."""
        seen = set()
        confirmed = []
        for face in faces:
            track = face.get('track_id')
            if track is None:
                continue
            seen.add(track)
            bbox = face['bbox']
            if track in self.alerted:
                self.alerted[track] = (bbox, now)
                continue
            if face.get('authorized', False):
                self.pending.pop(track, None)
                continue
            if any(now - last <= self.rearm and _overlaps(bbox, old)
                   for old, last in self.alerted.values()):
                # 추적 번호만 바뀐 같은 사람
                self.alerted[track] = (bbox, now)
                self.pending.pop(track, None)
                continue
            started = self.pending.setdefault(track, now)
            if face.get('detector_hold', False):
                continue
            if now - started >= self.confirm:
                self.pending.pop(track, None)
                self.alerted[track] = (bbox, now)
                confirmed.append(face)

        for track in [t for t in self.pending if t not in seen]:
            del self.pending[track]
        for track in [t for t, (_, last) in self.alerted.items() if now - last > self.rearm]:
            del self.alerted[track]
        return confirmed
