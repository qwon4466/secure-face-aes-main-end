"""오래된 녹화 자동 삭제.

- 보관 기간(RETENTION_DAYS)이 지난 녹화 청크를 지운다.
- 남은 공간이 MIN_FREE_GB보다 적으면 보관 기간과 상관없이 가장 오래된 청크부터
  공간이 확보될 때까지 지운다. 1분 영상이 약 300MB라 기간 기준만으로는 SD카드가
  몇 시간 안에 가득 찬다.
- 방금 저장된 청크(protect_seconds 이내)와 저장 중인 임시 폴더(.으로 시작)는 지우지 않는다.
- 복원 결과(.results: 복호화된 원본 영상)는 result_keep_seconds가 지나면 지운다.
"""
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
HOUR = re.compile(r'^\d{2}$')
STAMP = re.compile(r'^\d{2}-\d{2}-\d{2}$')


def _started(day, hour, chunk):
    try:
        if STAMP.fullmatch(chunk.name):
            return datetime.strptime(f'{day.name} {chunk.name}', '%Y-%m-%d %H-%M-%S').timestamp()
        return datetime.strptime(f'{day.name} {hour.name}', '%Y-%m-%d %H').timestamp()
    except ValueError:
        return chunk.stat().st_mtime


def chunk_dirs(root):
    """root/YYYY-MM-DD/HH/<청크>를 (촬영 시각, 경로)로 오래된 순 반환."""
    root = Path(root)
    found = []
    if not root.is_dir():
        return found
    for day in root.iterdir():
        if day.is_symlink() or not day.is_dir() or not DATE.fullmatch(day.name):
            continue
        for hour in day.iterdir():
            if hour.is_symlink() or not hour.is_dir() or not HOUR.fullmatch(hour.name):
                continue
            for chunk in hour.iterdir():
                if chunk.name.startswith('.') or chunk.is_symlink() or not chunk.is_dir():
                    continue
                found.append((_started(day, hour, chunk), chunk))
    return sorted(found)


def _size(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def _remove(path, root):
    """root 안의 경로만 지우고, 비게 된 시간·날짜 폴더도 정리한다."""
    path, root = Path(path), Path(root).resolve()
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError(f'삭제 대상이 데이터 폴더 밖입니다: {path}')
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    parent = path.parent
    while parent.resolve() != root and parent.resolve().is_relative_to(root):
        try:
            parent.rmdir()   # 비어 있을 때만 지워진다
        except OSError:
            break
        parent = parent.parent


def cleanup(root, results=(), *, retention_days=7, min_free_bytes=5 * 1024**3,
            protect_seconds=600, result_keep_seconds=24 * 3600, now=None, disk_free=None):
    """한 번 정리하고 무엇을 지웠는지 돌려준다."""
    root = Path(root)
    now = time.time() if now is None else now
    disk_free = disk_free or (lambda: shutil.disk_usage(root).free)
    report = {'expired': [], 'space': [], 'results': [], 'freed_bytes': 0}

    def drop(path, base, kind):
        size = _size(path)
        _remove(path, base)
        report['freed_bytes'] += size
        report[kind].append(Path(path).relative_to(base).as_posix())

    def fresh(path):
        return now - Path(path).stat().st_mtime < protect_seconds

    remaining = []
    for started, path in chunk_dirs(root):
        if retention_days and now - started > retention_days * 86400 and not fresh(path):
            drop(path, root, 'expired')
        else:
            remaining.append(path)
    for path in remaining:   # 오래된 순
        if disk_free() >= min_free_bytes:
            break
        if not fresh(path):
            drop(path, root, 'space')

    for base in results:
        base = Path(base)
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if not entry.is_symlink() and now - entry.stat().st_mtime > result_keep_seconds:
                drop(entry, base, 'results')
    return report
