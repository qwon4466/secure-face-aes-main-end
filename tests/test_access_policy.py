import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

from core.access_policy import AccessEnv, evaluate, parse_floors, stored_floors

DAY = {'after_hours': False, 'floor': 2}
NIGHT = {'after_hours': True, 'floor': 2}


def test_evaluate_order_and_reasons():
    assert evaluate(None, DAY) == (False, '미등록')
    assert evaluate({'group': '비허가', 'floors': None}, DAY) == (False, '비허가 등록')
    assert evaluate({'group': '허가', 'floors': None}, DAY) == (True, '')
    assert evaluate({'group': '허가', 'floors': None}, NIGHT) == (False, '근무시간 외')
    assert evaluate({'group': '허가', 'floors': [1]}, DAY) == (False, '2층 출입금지')
    assert evaluate({'group': '허가', 'floors': [1, 2]}, DAY) == (True, '')
    # 층 정보가 없는 기존 등록자(그룹 정보만 있음)는 전체 층 허가
    assert evaluate({'group': '허가'}, DAY) == (True, '')


def test_parse_and_stored_floors():
    assert parse_floors([3, 1, 1], [1, 2, 3]) == [1, 3]
    for bad in ([], [4], ['x'], None):
        with pytest.raises(ValueError):
            parse_floors(bad, [1, 2, 3])
    assert stored_floors('1, 3,9,x', [1, 2, 3]) == [1, 3]
    assert stored_floors(None, [1, 2, 3]) == []


def test_real_mode_uses_clock_and_camera_floor(work):
    env = AccessEnv(work / 'env.json', [1, 2, 3], camera_floor=2)
    tuesday = datetime(2026, 10, 20)
    assert env.effective(tuesday.replace(hour=10)) == {
        'demo': False, 'day': 1, 'time': '10:00', 'floor': 2, 'after_hours': False}
    assert env.effective(tuesday.replace(hour=18))['after_hours'] is True    # 18:00부터 근무시간 외
    assert env.effective(tuesday.replace(hour=8, minute=59))['after_hours'] is True
    assert env.effective(datetime(2026, 10, 24, 10))['after_hours'] is True   # 토요일


def test_demo_mode_overrides_and_persists(work):
    path = work / 'access_env.json'
    env = AccessEnv(path, [1, 2, 3])
    env.update(demo=True, day=1, time='17:59', floor=3)
    assert env.effective() == {'demo': True, 'day': 1, 'time': '17:59', 'floor': 3, 'after_hours': False}
    env.update(time='19:00')
    assert env.effective()['after_hours'] is True
    assert json.loads(path.read_text()) == {'demo': True, 'day': 1, 'time': '19:00', 'floor': 3}
    for bad in ({'floor': 7}, {'day': 7}, {'time': '25:00'}, {'time': ''}):
        with pytest.raises(ValueError):
            env.update(**bad)
    assert AccessEnv(path, [1, 2, 3]).state == {'demo': True, 'day': 1, 'time': '19:00', 'floor': 3}
    env.reset()
    assert AccessEnv(path, [1, 2, 3]).state['demo'] is False
    path.write_text('{broken')
    assert AccessEnv(path, [1, 2, 3]).state == AccessEnv(work / 'none.json', [1, 2, 3]).defaults()


def _camera(work, env_state, people):
    from core.selective_camera import MainCamera
    env = AccessEnv(work / 'env.json', [1, 2, 3])
    env.update(**env_state)
    cam = SimpleNamespace(access_env=env, people=people, access_floors=[1, 2, 3])
    return MainCamera, cam


def test_apply_access_changes_display_only_never_encryption_group(work):
    MainCamera, cam = _camera(work, {'demo': True, 'day': 1, 'time': '10:00', 'floor': 2},
                              {'kim': {'group': '허가', 'floors': [1]},
                               'lee': {'group': '허가', 'floors': None},
                               'park': {'group': '비허가', 'floors': None}})
    faces = [{'track_id': t, 'bbox': [0, 0, 10, 10], 'authorized': True, 'group': 'internal'}
             for t in (1, 2, 3)]
    faces.append({'track_id': 4, 'bbox': [0, 0, 10, 10], 'authorized': False, 'group': 'external'})
    states = {1: {'identity': 'kim'}, 2: {'identity': 'lee'}, 3: {'identity': 'park'}, 4: {'identity': None}}
    names = MainCamera._apply_access(cam, faces, states)
    assert [(f['allowed'], f['access_reason']) for f in faces] == [
        (False, '2층 출입금지'), (True, ''), (False, '비허가 등록'), (False, '미등록')]
    assert [f['group'] for f in faces] == ['internal', 'internal', 'internal', 'external']
    assert names == {1: 'kim', 2: 'lee', 3: 'park'}
    assert all('kim' not in json.dumps(f) for f in faces)   # 이름은 얼굴 기록(메타데이터)에 넣지 않음

    cam.access_env.update(time='19:00')
    MainCamera._apply_access(cam, faces, states)
    assert [f['access_reason'] for f in faces] == ['근무시간 외', '근무시간 외', '비허가 등록', '미등록']


def test_load_people_uses_latest_group_and_floors(work):
    from core.selective_camera import MainCamera
    db = work / 'users.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, auth_group TEXT)')
    conn.execute('CREATE TABLE person_access (name TEXT PRIMARY KEY, floors TEXT NOT NULL)')
    conn.executemany('INSERT INTO users (name, auth_group) VALUES (?, ?)',
                     [('kim_정면', '허가'), ('kim_연속_3', '비허가'), ('lee_정면', '허가')])
    conn.execute("INSERT INTO person_access VALUES ('kim', '1,3')")
    conn.commit(); conn.close()
    cam = SimpleNamespace(gallery_path=str(db), access_floors=[1, 2, 3])
    MainCamera._load_people(cam)
    assert cam.people == {'kim': {'group': '비허가', 'floors': [1, 3]},
                          'lee': {'group': '허가', 'floors': None}}
