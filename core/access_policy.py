"""시간·층별 출입 판정 (화면 표시와 알림 전용).

암호화 그룹(face['group'])은 바꾸지 않는다. 등록자로 인식된 얼굴은 출입이 거부되어도
내부인 그룹으로 저장되므로, 나중에 내부인 비밀번호로 복원할 수 있다.

관제 환경은 실제 시각·카메라 설치 층을 따르고, 대시보드에서 시연 모드를 켜면
가정한 요일·시간·층을 대신 적용한다. 설정은 데이터 폴더에 저장한다.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

UNREGISTERED = '미등록'
DENIED_GROUP = '비허가 등록'
AFTER_HOURS = '근무시간 외'
DAYS = ('월', '화', '수', '목', '금', '토', '일')
_HHMM = re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d$')


def parse_floors(value, valid):
    """등록 요청의 층 목록 검사. 비어 있으면 오류."""
    try:
        floors = sorted({int(f) for f in value})
    except (TypeError, ValueError):
        raise ValueError('층 목록 형식 오류')
    if not floors:
        raise ValueError('출입 허가 층을 하나 이상 선택하세요')
    if any(f not in valid for f in floors):
        raise ValueError('없는 층이 포함되어 있습니다')
    return floors


def stored_floors(text, valid):
    """DB에 저장된 '1,3' → [1, 3]. 지금 설정에 없는 층은 무시한다."""
    floors = []
    for item in str(text or '').split(','):
        item = item.strip()
        if item.isdigit() and int(item) in valid:
            floors.append(int(item))
    return sorted(set(floors))


def evaluate(person, env):
    """(허가 여부, 사유). person은 미인식이면 None, 등록자면 {'group', 'floors'(없으면 전체 층)}."""
    if person is None:
        return False, UNREGISTERED
    if person.get('group') == '비허가':
        return False, DENIED_GROUP
    if env.get('after_hours'):
        return False, AFTER_HOURS
    floors = person.get('floors')
    if floors is not None and env.get('floor') not in floors:
        return False, f"{env.get('floor')}층 출입금지"
    return True, ''


class AccessEnv:
    """관제 환경. state는 통째로 바꿔 끼워 카메라 스레드가 잠금 없이 읽는다."""

    def __init__(self, path, floors, camera_floor=None, work_days=(0, 1, 2, 3, 4),
                 work_start='09:00', work_end='18:00'):
        self.path = Path(path)
        self.floors = sorted({int(f) for f in floors})
        self.camera_floor = int(camera_floor) if camera_floor in self.floors else self.floors[0]
        self.work_days = tuple(sorted({int(d) for d in work_days}))
        if not (_HHMM.match(work_start) and _HHMM.match(work_end) and work_start < work_end):
            raise ValueError('근무 시간 설정 오류')
        self.work_start, self.work_end = work_start, work_end
        self.state = self.defaults()
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            self.update(**{k: saved.get(k) for k in ('demo', 'day', 'time', 'floor')}, save=False)
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def defaults(self):
        return {'demo': False, 'day': 1, 'time': '19:00', 'floor': self.camera_floor}

    def effective(self, now=None):
        """지금 판정에 쓰는 환경. 시연 모드면 가정 값, 아니면 실제 시각과 카메라 설치 층."""
        state = self.state
        if state['demo']:
            day, hhmm, floor = state['day'], state['time'], state['floor']
        else:
            now = now or datetime.now()
            day, hhmm, floor = now.weekday(), now.strftime('%H:%M'), self.camera_floor
        after = day not in self.work_days or not (self.work_start <= hhmm < self.work_end)
        return {'demo': state['demo'], 'day': day, 'time': hhmm, 'floor': floor, 'after_hours': after}

    def get(self):
        return dict(self.state, floors=list(self.floors), camera_floor=self.camera_floor,
                    work={'days': list(self.work_days), 'start': self.work_start, 'end': self.work_end},
                    effective=self.effective())

    def update(self, demo=None, day=None, time=None, floor=None, save=True):
        state = dict(self.state)
        if demo is not None:
            state['demo'] = bool(demo)
        if day is not None:
            day = int(day)
            if not 0 <= day <= 6:
                raise ValueError('요일 오류')
            state['day'] = day
        if time is not None:
            if not isinstance(time, str) or not _HHMM.match(time):
                raise ValueError('시간은 HH:MM 형식이어야 합니다')
            state['time'] = time
        if floor is not None:
            floor = int(floor)
            if floor not in self.floors:
                raise ValueError('없는 층입니다')
            state['floor'] = floor
        if save:
            tmp = self.path.with_name(self.path.name + '.tmp')
            tmp.write_text(json.dumps(state), encoding='utf-8')
            os.replace(tmp, self.path)
        self.state = state
        return self.get()

    def reset(self):
        self.state = self.defaults()
        return self.update()
