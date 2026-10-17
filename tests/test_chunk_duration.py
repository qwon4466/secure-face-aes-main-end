from pathlib import Path
import secrets
import threading
import time
import numpy as np
import pytest
from core import selective_recording
from core.selective_crypto import GROUPS,RestoreError,inspect_chunk,keyset,generate_keys,restore_frames
from core.selective_recording import Recorder
from core.selective_index import completed,find,summary

def keys(work):
    p=work/'keys';generate_keys(p);return keyset(p)

def test_duration_chunk_timed_path_and_short_tail_not_published(work):
    recorder=Recorder(work/'raw_data',keys(work),fps=2,chunk_frames=8,duration=1)
    frame=np.random.default_rng(3).integers(0,255,(24,32,3),dtype=np.uint8)
    recorder.add(frame,[],0,0);recorder.chunk_start-=1.01
    recorder.add(frame,[],1,1010) # closes the preceding one-second segment
    rows=completed(work/'raw_data');assert len(rows)==1
    path,meta=rows[0]
    assert path.relative_to(work/'raw_data').as_posix().count('/')==2
    assert path.name.count('-')==2 and meta['duration_seconds']==1
    assert find(work/'raw_data',meta['chunk'])==path
    item=summary(path,meta,work/'raw_data');assert item['complete'] and item['frames']==1
    recorder.finish() # second, short segment is discarded
    assert len(completed(work/'raw_data'))==1
    assert not list((work/'raw_data').rglob('*.partial'))

def test_invalid_duration_rejected(work):
    for value in (0,-1,False):
        with pytest.raises(RestoreError):Recorder(work/str(value),keys(work/str(value)),duration=value,fps=2)

def test_partial_directory_is_not_indexed(work):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from core.selective_crypto import save_chunk
    frame=np.zeros((24,32,3),np.uint8);root=work/'raw_data';final=root/'2026-09-16/21/21-34-56'
    with pytest.raises(OSError):
        save_chunk(root,[frame],[[]],[{'capture_id':0,'timestamp_ms':0}],keys(work),
                   '1'*32,fps=1,final_path=final,duration=1,fail_at='video',
                   started_at=datetime.now(ZoneInfo('Asia/Seoul')),ended_at=datetime.now(ZoneInfo('Asia/Seoul')))
    assert (final.parent/'.21-34-56.partial').is_dir()
    assert completed(root)==[] and not final.exists()


def test_background_chunk_save_does_not_block_next_frame(work,monkeypatch):
    started=threading.Event();release=threading.Event()
    def slow_save(output,*args,**kwargs):
        started.set();assert release.wait(2)
        return Path(output)/'saved-chunk'
    monkeypatch.setattr(selective_recording,'save_chunk',slow_save)
    recorder=Recorder(work/'raw_data',keys(work),fps=10,chunk_frames=2,
                      background_flush=True)
    frame=np.zeros((24,32,3),np.uint8)
    recorder.add(frame,[],0,0)
    before=time.perf_counter();recorder.add(frame,[],1,100)
    assert time.perf_counter()-before<0.1 and started.wait(1)
    recorder.add(frame,[],2,200)
    release.set();report=recorder.finish()
    assert report['frames']==3 and len(report['chunks'])==2
