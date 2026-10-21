import io
import sqlite3

import numpy as np

from core import gallery_model

CPU, HAILO = 'onnx:w600k_mbf.onnx', 'hailo:arcface_mobilefacenet.hef'


class FakeDetector:
    def detect(self, image, max_num=0, metric='default'):
        if image == 'no-face':
            return np.zeros((0, 5)), None
        # 작은 얼굴 하나, 큰 얼굴 하나: 큰 얼굴(등록 대상)을 골라야 한다
        return (np.array([[0, 0, 10, 10, .9], [0, 0, 50, 50, .9]]),
                np.array([np.full((5, 2), 1.0), np.full((5, 2), 2.0)]))


class FakeRecognizer:
    def get_feat(self, aligned):
        return np.full(512, aligned, dtype=np.float32)


def align(image, landmark, image_size):
    return float(landmark[0][0])          # 고른 얼굴의 랜드마크 값이 벡터가 된다


def make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, '
                 'auth_group TEXT, image_path TEXT, vector BLOB)')
    for name, image in rows:
        conn.execute('INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)',
                     (name, '허가', image, gallery_model._blob(np.zeros(512))))
    conn.commit(); conn.close()


def vectors(path):
    conn = sqlite3.connect(path)
    rows = conn.execute('SELECT name, vector FROM users ORDER BY id').fetchall()
    conn.close()
    return {n: float(np.load(io.BytesIO(v))[0]) for n, v in rows}


def sync(path, model, logs=None):
    images = {'kim.jpg': 'face', 'lee.jpg': 'face', 'empty.jpg': 'no-face'}
    return gallery_model.sync(path, model, CPU, FakeDetector(), FakeRecognizer(), align,
                              images.get, log=(logs.append if logs is not None else lambda m: None))


def test_switch_to_hailo_reembeds_from_photos_with_backup(work):
    db = work / 'users.db'
    make_db(db, [('kim_정면', 'kim.jpg'), ('lee_정면', 'lee.jpg'),
                 ('park_정면', 'missing.jpg'), ('choi_정면', 'empty.jpg')])
    logs = []
    report = sync(db, HAILO, logs)
    assert report['updated'] == ['kim_정면', 'lee_정면']
    assert sorted(report['removed']) == ['choi_정면', 'park_정면']   # 사진 없음 / 얼굴 없음 → 삭제
    assert vectors(db) == {'kim_정면': 2.0, 'lee_정면': 2.0}          # 가장 큰 얼굴로 계산
    backup = work / report['backup'].split('\\')[-1].split('/')[-1]
    assert backup.exists() and len(vectors(backup)) == 4             # 백업은 원래 그대로
    assert any('다시 등록 필요' in m for m in logs)
    assert sync(db, HAILO) is None                                   # 두 번째부터는 할 일 없음


def test_same_model_or_empty_db_only_records_model(work):
    db = work / 'users.db'
    make_db(db, [('kim_정면', 'kim.jpg')])
    assert sync(db, CPU) is None                                     # 기록 없는 기존 DB = CPU 벡터
    assert vectors(db) == {'kim_정면': 0.0}
    conn = sqlite3.connect(db)
    assert gallery_model.stored_model(conn) == CPU
    conn.close()
    empty = work / 'empty.db'
    make_db(empty, [])
    assert sync(empty, HAILO) is None
    assert not list(work.glob('empty.db.bak-*'))


def test_switching_back_to_cpu_reembeds_again(work):
    db = work / 'users.db'
    make_db(db, [('kim_정면', 'kim.jpg')])
    sync(db, HAILO)
    report = sync(db, CPU)                                           # Hailo 실패로 CPU 폴백 등
    assert report['updated'] == ['kim_정면']


def test_missing_db_is_ignored(work):
    assert sync(work / 'none.db', HAILO) is None
