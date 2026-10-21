"""
이벤트/감사 로그 (JSON Lines).

- 복원 접근 감사(restore_audit.jsonl): 누가/언제/어떤 청크를 복원했는지 기록.
- 비허가자 감지 이벤트(detect_events.jsonl): 미등록 인원 감지 시 기록.

프라이버시 시스템의 책임성(accountability)을 위해, 익명화 해제(복원)는 반드시
기록으로 남긴다. main.py가 복원 시와 카메라의 비허가자 감지 이벤트 시 사용한다.
"""
import json
import os
import threading
from datetime import datetime

_LOG_DIR = "logs"
_RESTORE_LOG = os.path.join(_LOG_DIR, "restore_audit.jsonl")
_DETECT_LOG = os.path.join(_LOG_DIR, "detect_events.jsonl")
_lock = threading.Lock()


def set_log_dir(path):
    """기록 위치 변경 (테스트 실행이 운영 기록에 섞이지 않도록)."""
    global _LOG_DIR, _RESTORE_LOG, _DETECT_LOG
    _LOG_DIR = str(path)
    _RESTORE_LOG = os.path.join(_LOG_DIR, "restore_audit.jsonl")
    _DETECT_LOG = os.path.join(_LOG_DIR, "detect_events.jsonl")


def _append(path: str, entry: dict):
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        entry["time"] = datetime.now().isoformat(timespec="seconds")
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[event_log] 기록 실패: {e}")


def log_restore(chunk_id: str, who: str, ip: str, success: bool, mode: str = "gpu"):
    """복원(익명화 해제) 접근 기록."""
    _append(_RESTORE_LOG, {
        "chunk_id": chunk_id, "who": who or "미로그인",
        "ip": ip or "-", "success": bool(success), "mode": mode,
    })


def log_detection(name: str, group: str, sim, track=None, reason=None):
    """비허가자 감지 기록. sim은 인식 실패 시 None, reason은 출입 거부 사유."""
    _append(_DETECT_LOG, {
        "name": name, "group": group,
        "sim": None if sim is None else round(float(sim), 3),
        "track": track, "reason": reason,
    })


def _read(path: str, limit: int) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out = []
    for ln in lines[-limit:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    out.reverse()   # 최신순
    return out


def read_restore(limit: int = 200) -> list:
    return _read(_RESTORE_LOG, limit)


def read_detections(limit: int = 200) -> list:
    return _read(_DETECT_LOG, limit)


def count_detections() -> int:
    """비허가자 감지 이벤트 총 개수 (경고 카운트용)."""
    if not os.path.exists(_DETECT_LOG):
        return 0
    try:
        with open(_DETECT_LOG, encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0
