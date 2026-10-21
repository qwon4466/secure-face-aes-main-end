import os
from datetime import datetime

import pytest

from core.retention import chunk_dirs, cleanup

NOW = datetime(2026, 10, 21, 12, 0, 0).timestamp()
GB = 1024**3


def make_chunk(root, stamp, size=1000, mtime=None):
    """stamp: 'YYYY-MM-DD HH:MM:SS' → root/YYYY-MM-DD/HH/HH-MM-SS/ (파일 하나)."""
    day, clock = stamp.split()
    path = root / day / clock[:2] / clock.replace(':', '-')
    path.mkdir(parents=True)
    (path / 'raw.bin').write_bytes(b'x' * size)
    t = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S').timestamp() + 60 if mtime is None else mtime
    os.utime(path, (t, t))
    return path


def run(root, **kw):
    kw.setdefault('disk_free', lambda: 100 * GB)
    return cleanup(root, now=NOW, **kw)


def test_expired_chunks_deleted_and_empty_folders_removed(work):
    old = make_chunk(work, '2026-10-10 09:00:00')
    keep = make_chunk(work, '2026-10-20 09:00:00')
    report = run(work, retention_days=7)
    assert report['expired'] == ['2026-10-10/09/09-00-00']
    assert not old.exists() and not (work / '2026-10-10').exists()   # 빈 날짜 폴더도 정리
    assert keep.exists() and report['space'] == [] and report['freed_bytes'] == 1000


def test_low_disk_deletes_oldest_first_until_enough_space(work):
    chunks = [make_chunk(work, f'2026-10-21 10:0{i}:00', size=GB // 4) for i in range(5)]
    deleted = []
    free = {'bytes': 4.6 * GB}

    def disk_free():
        return free['bytes']
    real_rmtree = __import__('shutil').rmtree

    def tracking_rmtree(path, *a, **k):
        deleted.append(path); free['bytes'] += GB / 4; real_rmtree(path, *a, **k)
    import core.retention as retention
    retention.shutil.rmtree, original = tracking_rmtree, retention.shutil.rmtree
    try:
        report = run(work, retention_days=7, min_free_bytes=5 * GB, disk_free=disk_free)
    finally:
        retention.shutil.rmtree = original
    assert report['space'] == ['2026-10-21/10/10-00-00', '2026-10-21/10/10-01-00']   # 오래된 순, 필요한 만큼만
    assert [c.exists() for c in chunks] == [False, False, True, True, True]


def test_fresh_and_in_progress_chunks_never_deleted(work):
    fresh = make_chunk(work, '2026-10-21 11:55:00', mtime=NOW - 60)          # 방금 저장
    partial = work / '2026-10-21' / '11' / '.11-59-00.partial'               # 저장 중
    partial.mkdir(parents=True)
    report = run(work, retention_days=7, min_free_bytes=5 * GB, disk_free=lambda: 1 * GB)
    assert report['space'] == [] and fresh.exists() and partial.exists()
    assert [p for _, p in chunk_dirs(work)] == [fresh]


def test_restore_results_expire(work):
    results = work / '.results'
    old = results / 'job-old'; old.mkdir(parents=True); (old / 'restored.mp4').write_bytes(b'v')
    new = results / 'original-new.mp4'; new.write_bytes(b'v')
    os.utime(old, (NOW - 25 * 3600,) * 2)
    os.utime(new, (NOW - 3600,) * 2)
    report = run(work, results=[results], result_keep_seconds=24 * 3600)
    assert report['results'] == ['job-old'] and not old.exists() and new.exists() and results.exists()


def test_zero_retention_days_disables_age_rule_and_other_folders_untouched(work):
    make_chunk(work, '2020-01-01 00:00:00')
    (work / '.keys').mkdir(); (work / 'access_env.json').write_text('{}')
    (work / 'not-a-date').mkdir()
    report = run(work, retention_days=0)
    assert report['expired'] == []
    assert (work / '.keys').exists() and (work / 'access_env.json').exists() and (work / 'not-a-date').exists()
