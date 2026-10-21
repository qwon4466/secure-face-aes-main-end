"""얼굴 DB 벡터와 인식 모델 맞추기.

ArcFace 모델(CPU ONNX ↔ Hailo-8L HEF)이 바뀌면 같은 얼굴이라도 벡터 공간이 달라져
기존 등록 벡터로는 아무도 인식되지 않는다. 그래서 DB에 어떤 모델로 만든 벡터인지
기록해 두고, 모델이 바뀌면 저장된 등록 사진(users.image_path)으로 벡터를 다시 계산한다.

- 다시 계산하기 전에 DB를 통째로 백업한다 (<db>.bak-<시각>).
- 사진이 없거나 얼굴을 찾지 못한 행은 지운다. 다른 모델의 벡터가 남아 있으면 엉뚱한 사람과
  일치할 수 있기 때문이다. 지운 이름은 보고서에 남기고 다시 등록하도록 안내한다.
"""
import io
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

_META = 'CREATE TABLE IF NOT EXISTS gallery_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)'


def _blob(vec):
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32), allow_pickle=False)
    return sqlite3.Binary(buf.getvalue())


def stored_model(conn):
    conn.execute(_META)
    row = conn.execute("SELECT value FROM gallery_meta WHERE key = 'recognizer'").fetchone()
    return row[0] if row else None


def set_model(conn, model_id):
    conn.execute(_META)
    conn.execute("INSERT INTO gallery_meta (key, value) VALUES ('recognizer', ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (model_id,))


def _embed(image, detector, recognizer, align):
    """등록 사진에서 가장 큰 얼굴 하나의 벡터. 얼굴이 없으면 None."""
    boxes, kpss = detector.detect(image, max_num=0, metric='default')
    if boxes is None or len(boxes) == 0 or kpss is None:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    i = int(np.argmax(areas))
    emb = recognizer.get_feat(align(image, landmark=kpss[i], image_size=112))
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb if np.isfinite(emb).all() and emb.size in (128, 256, 512) else None


def sync(db_path, model_id, legacy_model_id, detector, recognizer, align, load_image, log=print):
    """DB 벡터가 model_id로 만든 것인지 확인하고, 아니면 다시 계산한다.

    legacy_model_id: 기록이 없는 기존 DB의 벡터를 만든 모델 (이 기능 이전에는 CPU ONNX).
    반환: None(할 일 없음) 또는 {'updated', 'removed', 'backup'}.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(db_path)
    try:
        current = stored_model(conn) or legacy_model_id
        if current == model_id:
            set_model(conn, model_id)
            conn.commit()
            return None
        rows = conn.execute('SELECT id, name, image_path FROM users ORDER BY id').fetchall()
        if not rows:
            set_model(conn, model_id)
            conn.commit()
            return None
        backup = db_path.with_name(f'{db_path.name}.bak-{time.strftime("%Y%m%d-%H%M%S")}')
        conn.commit()
        shutil.copy2(db_path, backup)
        log(f'[GALLERY] 인식 모델 변경 {current} → {model_id} | 등록 사진으로 벡터 재계산 시작 '
            f'({len(rows)}개) | 백업={backup.name}')
        updated, removed = [], []
        for row_id, name, image_path in rows:
            image = load_image(str(image_path)) if image_path else None
            emb = _embed(image, detector, recognizer, align) if image is not None else None
            if emb is None:
                conn.execute('DELETE FROM users WHERE id = ?', (row_id,))
                removed.append(name)
            else:
                conn.execute('UPDATE users SET vector = ? WHERE id = ?', (_blob(emb), row_id))
                updated.append(name)
        set_model(conn, model_id)
        conn.commit()
        log(f'[GALLERY] 재계산 완료 | 갱신={len(updated)}개 | 삭제={len(removed)}개'
            + (f' | 다시 등록 필요: {", ".join(sorted(set(removed)))}' if removed else ''))
        return {'updated': updated, 'removed': removed, 'backup': str(backup)}
    finally:
        conn.close()
