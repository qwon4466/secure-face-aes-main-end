"""Version 1: lossless BGR video + sparse, cryptographically separated pixels."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import uuid
import cv2
import numpy as np
from types import SimpleNamespace
import sys
display_config = sys.modules.get("config") or SimpleNamespace(SELECTIVE_BLUR_MIN_KERNEL=61, SELECTIVE_BLUR_FACE_RATIO=0.9, SELECTIVE_BLUR_WORK_SIZE=32, SELECTIVE_BLUR_EXPANSION_RATIO=0.20)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.keywrap import aes_key_wrap, aes_key_unwrap, InvalidUnwrap

GROUPS = ('external', 'internal', 'master')
RAW_NAME = 'encrypted_raw.bin.enc'
RAW_META_NAME = 'encrypted_raw.bin.json'
ROOT = Path(__file__).resolve().parents[1]
MAX_FRAMES = 16
MAX_CHUNK_FRAMES = 1024
MAX_PIXELS = 1920 * 1080
MAX_RAW = 64 * 1024 * 1024
MAX_CHUNK_RAW = 1024 * 1024 * 1024
MAX_FILE = 1024 * 1024 * 1024
RADIUS = 15
GUARD = 2
ID = re.compile(r'^[0-9a-f]{32}$')

class RestoreError(ValueError):
    pass

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def b64(data):
    return base64.b64encode(data).decode('ascii')

def unb64(value):
    return base64.b64decode(value, validate=True)

def private_dir(path):
    path = Path(path).absolute()
    # All newly written data stays inside private application data or test artifacts; never follow symlink output parents.
    if not any(path.resolve().is_relative_to(p) for p in (ROOT/'raw_data', ROOT/'tests/runtime')) or any(p.is_symlink() for p in (path, *path.parents)):
        raise RestoreError('출력은 선택 복원 데이터 폴더 내부의 심볼릭 링크가 아닌 경로여야 합니다')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path

def write_new(path, data):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def chunk_document(chunk):
    """Read the existing seven-file format or the two-file container index."""
    chunk=Path(chunk)
    if (chunk/'manifest.json').is_file():
        return json.loads(read_bounded(chunk/'manifest.json',16*1024*1024),
                          object_pairs_hook=unique_json),None,None
    index=json.loads(read_bounded(chunk/RAW_META_NAME,16*1024*1024),
                     object_pairs_hook=unique_json)
    if index.get('format')!='srx-container-v2' or index.get('status')!='completed':
        raise RestoreError('완료된 청크 인덱스가 아닙니다')
    sections=index['sections']
    if set(sections)!={'original','protected',*GROUPS}:
        raise RestoreError('청크 레코드 목록 오류')
    length=(chunk/RAW_NAME).stat().st_size
    offset=0
    for name in ('original','protected',*GROUPS):
        section=sections[name]
        if set(section)!={'offset','length','sha256'} or type(section['offset']) is not int or type(section['length']) is not int or section['offset']!=offset or section['length']<0 or not re.fullmatch('[0-9a-f]{64}',section['sha256']):
            raise RestoreError('청크 레코드 범위 오류')
        offset+=section['length']
    if offset!=length or length>MAX_FILE*3:
        raise RestoreError('청크 컨테이너 길이 오류')
    return index['document'],index['raw'],sections

def section_bytes(chunk,name,limit=MAX_FILE):
    chunk=Path(chunk)
    _,_,sections=chunk_document(chunk)
    if sections is None:
        filename=RAW_NAME if name=='original' else ('protected.avi' if name=='protected' else name+'.gcm')
        return read_bounded(chunk/filename,limit)
    section=sections[name]
    if section['length']>limit:
        raise RestoreError('청크 레코드 크기 상한 초과')
    with (chunk/RAW_NAME).open('rb') as source:
        source.seek(section['offset'])
        data=source.read(section['length'])
    if len(data)!=section['length'] or digest(data)!=section['sha256']:
        raise RestoreError('청크 레코드 무결성 오류')
    return data

def pack_chunk(stage, raw_metadata, document):
    """Append existing protected and group payloads to the original ciphertext."""
    container=stage/RAW_NAME
    sections={'original':{'offset':0,'length':container.stat().st_size,
                          'sha256':raw_metadata['ciphertext_sha256']}}
    with container.open('ab') as target:
        for name,filename in [('protected','protected.avi'),*((g,g+'.gcm') for g in GROUPS)]:
            source=stage/filename
            length=source.stat().st_size
            sections[name]={'offset':target.tell(),'length':length,'sha256':file_digest(source)}
            with source.open('rb') as inp:
                for block in iter(lambda:inp.read(1024*1024),b''):
                    target.write(block)
        target.flush();os.fsync(target.fileno())
    index={'format':'srx-container-v2','status':'completed','raw':raw_metadata,
           'document':document,'sections':sections}
    temp=stage/'.index.tmp'
    write_new(temp,canonical(index))
    os.replace(temp,stage/RAW_META_NAME)
    for name in ('protected.avi',*(g+'.gcm' for g in GROUPS),'manifest.json'):
        (stage/name).unlink()

def encrypt_original(stage, frames, master_key, fps, started_at, ended_at):
    """Stream full unblurred BGR frames into a distinct AES-256-GCM payload."""
    nonce = secrets.token_bytes(12)
    h,w = frames[0].shape[:2]
    header = {'version':1, 'algorithm':'AES-256-GCM', 'nonce':b64(nonce),
              'aad':'secureface-rx/raw-bgr24/v1', 'encrypted_file':RAW_NAME,
              'frame_count':len(frames), 'fps':fps, 'width':w, 'height':h,
              'channels':3, 'started_at':started_at.isoformat() if started_at else '',
              'ended_at':ended_at.isoformat() if ended_at else ''}
    aad = canonical(header)
    cipher = Cipher(algorithms.AES(master_key), modes.GCM(nonce)).encryptor()
    cipher.authenticate_additional_data(aad)
    path = stage/RAW_NAME
    fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(fd,'wb') as target:
        for frame in frames:
            target.write(cipher.update(frame.tobytes()))
        target.write(cipher.finalize())
        target.flush(); os.fsync(target.fileno())
    metadata = header | {'tag':b64(cipher.tag), 'ciphertext_sha256':file_digest(path)}
    write_new(stage/RAW_META_NAME, canonical(metadata))
    print(f'[AES] 원본 청크 암호화 완료 | 암호문={RAW_NAME} | 메타데이터={RAW_META_NAME}',flush=True)
    return metadata

def validate_original(chunk, meta):
    _,embedded,sections=chunk_document(chunk)
    data = embedded if embedded is not None else json.loads(read_bounded(chunk/RAW_META_NAME,4096), object_pairs_hook=unique_json)
    fields={'version','algorithm','nonce','aad','encrypted_file','frame_count','fps',
            'width','height','channels','started_at','ended_at','tag','ciphertext_sha256'}
    if set(data)!=fields or data['version']!=1 or data['algorithm']!='AES-256-GCM' or data['aad']!='secureface-rx/raw-bgr24/v1' or data['encrypted_file']!=RAW_NAME or data['channels']!=3:
        raise RestoreError('원본 AES 메타데이터 형식 오류')
    if len(unb64(data['nonce']))!=12 or len(unb64(data['tag']))!=16 or not re.fullmatch('[0-9a-f]{64}',data['ciphertext_sha256']):
        raise RestoreError('원본 AES nonce/tag/해시 오류')
    shape=meta['shape']; count=len(meta['frames'])
    if (data['frame_count']!=count or data['width']!=shape[1] or data['height']!=shape[0]
            or data['fps']!=meta['fps'] or data['started_at']!=meta['started_at']
            or data['ended_at']!=meta['ended_at']):
        raise RestoreError('원본 AES 프레임 메타데이터 불일치')
    path=chunk/RAW_NAME
    raw_length=sections['original']['length'] if sections is not None else path.stat().st_size
    if path.is_symlink() or not path.is_file() or raw_length!=math.prod(shape)*count:
        raise RestoreError('원본 AES 암호문 누락 또는 길이 오류')
    if digest(section_bytes(chunk,'original'))!=data['ciphertext_sha256'] or digest(canonical(data))!=meta['raw_metadata_sha256']:
        raise RestoreError('원본 AES 암호문 또는 메타데이터 변조')
    return data

def decrypt_original(chunk, data, key):
    """Authenticate the entire ciphertext in anonymous memory before returning frames."""
    if len(key)!=32:raise RestoreError('키는 32바이트여야 합니다')
    header={k:v for k,v in data.items() if k not in ('tag','ciphertext_sha256')}
    decryptor=Cipher(algorithms.AES(key),modes.GCM(unb64(data['nonce']),unb64(data['tag']))).decryptor()
    decryptor.authenticate_additional_data(canonical(header))
    _,_,sections=chunk_document(chunk)
    remaining=sections['original']['length'] if sections is not None else (Path(chunk)/RAW_NAME).stat().st_size
    fd=os.memfd_create('srx-original',os.MFD_CLOEXEC)
    try:
        with os.fdopen(fd,'w+b') as memory:
            with open(Path(chunk)/RAW_NAME,'rb') as source:
                while remaining:
                    block=source.read(min(1024*1024,remaining))
                    if not block:raise RestoreError('원본 AES 암호문 단절')
                    remaining-=len(block)
                    memory.write(decryptor.update(block))
            memory.write(decryptor.finalize())
            memory.seek(0)
            size=data['width']*data['height']*3
            frames=[]
            for _ in range(data['frame_count']):
                raw=memory.read(size)
                if len(raw)!=size:raise RestoreError('원본 프레임 길이 오류')
                frames.append(np.frombuffer(raw,dtype=np.uint8).reshape(data['height'],data['width'],3).copy())
            return frames
    except Exception as exc:
        raise RestoreError('원본 AES-GCM 인증 또는 복호화 실패') from exc

def restore_original_frames(chunk, key):
    """Restore the complete unblurred stream with the authenticated master key."""
    doc,_=inspect_chunk(chunk)
    manifest=doc['manifest'];meta=manifest['meta']
    try:
        envelope=manifest['envelopes']['master']
        dek=aes_key_unwrap(key,unb64(envelope['wraps'][0]))
        AESGCM(dek).decrypt((1).to_bytes(12,'big'),unb64(doc['seals']['master']),canonical(manifest))
    except Exception as exc:
        raise RestoreError('전체키 인증 실패') from exc
    return decrypt_original(chunk,validate_original(Path(chunk),meta),key),meta

def load_key(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size != 32 or p.stat().st_mode & 0o077:
        raise RestoreError('키는 권한 0600인 32바이트 파일이어야 합니다')
    return p.read_bytes()

def generate_keys(path):
    path = Path(path)
    if path.exists():
        raise RestoreError('새 키 폴더를 지정하세요; 기존 키는 덮어쓰지 않습니다')
    private_dir(path)
    for group in GROUPS:
        write_new(path / (group + '.key'), secrets.token_bytes(32))
    return path

def keyset(path):
    keys = {g: load_key(Path(path) / (g + '.key')) for g in GROUPS}
    if len(set(keys.values())) != 3:
        raise RestoreError('집단키는 서로 달라야 합니다')
    return keys

def runs(mask):
    padded = np.pad(mask.ravel().astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [[int(a), int(b-a)] for a, b in zip(changes[::2], changes[1::2])]

def mask_from_runs(items, shape):
    n = math.prod(shape)
    if not isinstance(items, list) or len(items) > n:
        raise RestoreError('마스크 크기 오류')
    out = np.zeros(n, dtype=bool)
    end = 0
    for item in items:
        if not isinstance(item, list) or len(item) != 2:
            raise RestoreError('마스크 형식 오류')
        start, length = item
        if type(start) is not int or type(length) is not int or start < end or length <= 0 or start + length > n:
            raise RestoreError('마스크 범위 오류')
        out[start:start+length] = True
        end = start+length
    return out.reshape(shape)

def protect(frame, faces):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0]*frame.shape[1] > MAX_PIXELS:
        raise RestoreError('BGR uint8 / 최대 1920×1080 픽셀 입력이 필요합니다')
    h, w = frame.shape[:2]
    if len(faces) > 128:
        raise RestoreError('프레임당 최대 얼굴 128개')
    masks = {g: np.zeros((h, w), dtype=bool) for g in GROUPS}
    protected = frame.copy()
    records = []
    groups_present = set()
    # Blur each expanded ROI from the SAME original, never from already blurred frames.
    # Reduced ROI Gaussian blur bounds CPU cost; originals remain untouched.
    for face in faces:
        group = face.get('group', 'master')
        if group not in GROUPS:
            raise RestoreError('집단 오류')
        groups_present.add(group)
        box = face['bbox']
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise RestoreError('bbox 오류')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise RestoreError('bbox 범위 오류')
        # Blur, encrypted source mask and displayed box use the exact clamped detector bbox.
        a,b,c,d = x1,y1,x2,y2
        masks[group][b:d,a:c] = True
        roi=frame[b:d,a:c]
        kernel=max(display_config.SELECTIVE_BLUR_MIN_KERNEL,int(max(x2-x1,y2-y1)*display_config.SELECTIVE_BLUR_FACE_RATIO)) | 1
        scale=min(1.0,display_config.SELECTIVE_BLUR_WORK_SIZE/max(c-a,d-b))
        small=cv2.resize(roi,(max(1,round((c-a)*scale)),max(1,round((d-b)*scale))),interpolation=cv2.INTER_AREA)
        small_kernel=max(3,round(kernel*scale)) | 1
        blurred=cv2.GaussianBlur(small,(small_kernel,small_kernel),max(1,kernel*scale/3),borderType=cv2.BORDER_REFLECT_101)
        protected[b:d,a:c]=cv2.resize(blurred,(c-a,d-b),interpolation=cv2.INTER_LINEAR)
        records.append({**face, 'write_roi':[a,b,c,d]})
    union = np.logical_or.reduce(list(masks.values()))
    kernel = np.ones((GUARD*2+1, GUARD*2+1), np.uint8)
    exclusive = {}
    for g in GROUPS[:2]:
        if g not in groups_present or not groups_present.difference((g,)):
            # No pixels in this group, or no opposing mask to obscure them.
            exclusive[g] = masks[g]
        else:
            others = np.logical_or.reduce([masks[k] for k in GROUPS if k != g])
            forbidden = cv2.dilate(others.astype(np.uint8), kernel).astype(bool)
            exclusive[g] = masks[g] & ~forbidden
    exclusive['master'] = union & ~(exclusive['external'] | exclusive['internal'])
    if any("authorized" in face for face in faces):
        from .face_display import annotate
        protected=annotate(protected,records)
    return protected, exclusive, records

def write_video(path, frames, fps):
    path = Path(path)
    if path.exists() or not 0 < fps <= 120:
        raise RestoreError('영상 출력 경로 또는 FPS 오류')
    h,w = frames[0].shape[:2]
    if h % 2 or w % 2:
        raise RestoreError('FFV1 정확성을 위해 짝수 해상도가 필요합니다')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (w,h))
    if not writer.isOpened():
        raise RestoreError('FFV1 인코더를 열 수 없습니다')
    os.chmod(path, 0o600)
    try:
        for frame in frames:
            if frame.shape != frames[0].shape:
                raise RestoreError('청크 중 해상도 변경')
            writer.write(frame)
    finally:
        writer.release()
    decoded = read_video(path, len(frames), [h,w,3])
    if not all(np.array_equal(a,b) for a,b in zip(frames, decoded)):
        raise RestoreError('FFV1 BGR 왕복 검증 실패')
    with path.open('rb') as f:
        os.fsync(f.fileno())

def read_video(path, count, shape):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for _ in range(count):
            ok, frame = cap.read()
            if not ok or list(frame.shape) != shape:
                raise RestoreError('영상 프레임 대응 오류')
            frames.append(frame)
        if cap.read()[0]:
            raise RestoreError('영상 프레임 수 초과')
    finally:
        cap.release()
    return frames

def save_chunk(root, frames, faces, timing, keys, session, fps=10.0, fail_at=None, final_path=None, duration=None, drops=0, started_at=None, ended_at=None):
    if not ID.fullmatch(session) or not 1 <= len(frames) <= (MAX_CHUNK_FRAMES if final_path else MAX_FRAMES) or len(faces) != len(frames) or len(timing) != len(frames):
        raise RestoreError('청크 크기 / 세션 오류')
    if sum(f.nbytes for f in frames) > (MAX_CHUNK_RAW if final_path else MAX_RAW) or len(set(keys.values())) != 3 or any(len(k)!=32 for k in keys.values()):
        raise RestoreError('청크 메모리 상한 또는 키 오류')
    if any(f.shape != frames[0].shape for f in frames):
        raise RestoreError('청크 중 해상도 변경')
    previous = -1
    for clock in timing:
        if not isinstance(clock,dict) or type(clock.get('capture_id')) is not int or clock['capture_id'] <= previous or not isinstance(clock.get('timestamp_ms'),(int,float)) or not math.isfinite(clock['timestamp_ms']):
            raise RestoreError('촬영 ID / 시간 오류')
        previous = clock['capture_id']
    signer = Ed25519PrivateKey.generate()
    root = private_dir(root)
    chunk_id = uuid.uuid4().hex
    final = Path(final_path) if final_path else root/chunk_id
    if final.exists():raise RestoreError('청크 시작 시각 경로 충돌: '+str(final))
    private_dir(final.parent)
    stage = private_dir(final.parent / ('.'+final.name+'.partial' if final_path else '.partial-' + chunk_id))
    protected, raw = [], {g: [] for g in GROUPS}
    metadata = []
    for i,(frame, detections, clock) in enumerate(zip(frames, faces, timing)):
        p, masks, records = protect(frame, detections)
        protected.append(p)
        metadata.append({**clock, 'stored_index':i, 'pts':i, 'faces':records,
                         'masks':{g:runs(masks[g]) for g in GROUPS}})
        for g in GROUPS:
            raw[g].append(frame[masks[g]].tobytes())
    write_video(stage/'protected.avi', protected, fps)
    if fail_at == 'video':
        raise OSError('injected interrupted save')
    raw_metadata=encrypt_original(stage,frames,keys['master'],fps,started_at,ended_at)
    meta = {'signing_public':b64(signer.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)),
            'version':1,'session':session,'chunk':chunk_id,'complete':True,
            'raw_metadata_sha256':digest(canonical(raw_metadata)),
            'shape':list(frames[0].shape),'color':'BGR24','codec':'FFV1',
            'fps':fps,'duration_seconds':duration or len(frames)/fps,'dropped_frames':drops,
            'started_at':started_at.isoformat() if started_at else '',
            'ended_at':ended_at.isoformat() if ended_at else '',
            'time_base':[1,int(round(fps*1000))], 'pts_scale':1000,
            'blur':{'algorithm':'adaptive-reduced-gaussian','min_kernel':display_config.SELECTIVE_BLUR_MIN_KERNEL,'face_ratio':display_config.SELECTIVE_BLUR_FACE_RATIO,'work_size':display_config.SELECTIVE_BLUR_WORK_SIZE,'expansion_ratio':display_config.SELECTIVE_BLUR_EXPANSION_RATIO,'expansion':0,'guard':GUARD,'outside_write_halo':0},
            'frames':metadata,'protected_sha256':digest((stage/'protected.avi').read_bytes())}
    envelopes, deks = {}, {}
    for g in GROUPS:
        # Fresh DEK every attempt, one payload encryption per DEK. Never resume partial encryption.
        dek = AESGCM.generate_key(bit_length=256)
        deks[g] = dek
        nonce = (0).to_bytes(12, 'big')
        ct = AESGCM(dek).encrypt(nonce, b''.join(raw[g]), canonical({'meta':meta,'group':g}))
        write_new(stage/(g+'.gcm'), ct)
        recipients = (g,'master') if g != 'master' else ('master',)
        envelopes[g] = {'nonce':b64(nonce), 'sha256':digest(ct),
                        'wraps':[b64(aes_key_wrap(keys[k],dek)) for k in recipients]}
    manifest = {'meta':meta,'envelopes':envelopes}
    # A second, unique nonce authenticates the entire envelope index, including inaccessible groups.
    seals = {g:b64(AESGCM(deks[g]).encrypt((1).to_bytes(12,'big'),b'',canonical(manifest))) for g in GROUPS}
    document = {'manifest':manifest,'seals':seals}
    document['signature'] = b64(signer.sign(canonical(document)))
    write_new(stage/'manifest.json', canonical(document))
    if fail_at == 'manifest':
        raise OSError('injected interrupted save')
    validate_original(stage,meta)
    if final_path:
        pack_chunk(stage,raw_metadata,document)
    sync_dir(stage)
    if final.exists():raise RestoreError('청크 시작 시각 경로 충돌: '+str(final))
    os.rename(stage,final)
    sync_dir(final.parent)
    return final

def read_bounded(path, limit=MAX_FILE):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise RestoreError('파일 누락 / 크기 상한 / 링크 오류')
        data = f.read(limit + 1)
    if len(data) > limit:
        raise RestoreError('읽는 중 파일 크기 상한 초과')
    return data

def unique_json(pairs):
    out = {}
    for key,value in pairs:
        if key in out:
            raise RestoreError('중복 JSON 키')
        out[key] = value
    return out

def inspect_chunk(chunk):
    chunk = Path(chunk)
    timed = bool(re.fullmatch(r'\d{2}-\d{2}-\d{2}',chunk.name) and re.fullmatch(r'\d{2}',chunk.parent.name) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',chunk.parent.parent.name) and (chunk.resolve().is_relative_to(ROOT/'raw_data') or chunk.resolve().is_relative_to(ROOT/'tests/runtime')))
    if chunk.is_symlink() or any(p.is_symlink() for p in chunk.parents) or not (ID.fullmatch(chunk.name) or timed):
        raise RestoreError('완료된 청크 ID가 필요합니다')
    doc,_,sections=chunk_document(chunk)
    manifest = doc['manifest']; meta = manifest['meta']; shape = meta['shape']
    Ed25519PublicKey.from_public_bytes(unb64(meta['signing_public'])).verify(
        unb64(doc['signature']), canonical({'manifest':manifest,'seals':doc['seals']}))
    if meta['version'] != 1 or meta['complete'] is not True or (not timed and meta['chunk'] != chunk.name) or not ID.fullmatch(meta['session']):
        raise RestoreError('청크 식별자 / 버전 / 완료 상태 오류')
    if len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape) or shape[2]!=3 or shape[0]*shape[1]>MAX_PIXELS:
        raise RestoreError('해상도 상한 오류')
    count = len(meta['frames'])
    if not 1<=count<=(MAX_CHUNK_FRAMES if timed else MAX_FRAMES) or math.prod(shape)*count>(MAX_CHUNK_RAW if timed else MAX_RAW) or not 0<meta['fps']<=120 or meta['color']!='BGR24' or meta['codec']!='FFV1':
        raise RestoreError('프레임 / 메모리 / 포맷 상한 오류')
    masks=[]; previous=-1
    for i,f in enumerate(meta['frames']):
        if f['stored_index']!=i or f['pts']!=i or type(f['capture_id']) is not int or f['capture_id']<=previous or not math.isfinite(f['timestamp_ms']):
            raise RestoreError('프레임 시간 대응 오류')
        previous=f['capture_id']
        m = {g:mask_from_runs(f['masks'][g],shape[:2]) for g in GROUPS}
        if np.any(sum(x.astype(np.uint8) for x in m.values())>1):
            raise RestoreError('복원 마스크 중첩')
        masks.append(m)
    if set(manifest['envelopes'])!=set(GROUPS) or set(doc['seals'])!=set(GROUPS):
        raise RestoreError('집단 목록 오류')
    if digest(section_bytes(chunk,'protected')) != meta['protected_sha256']:
        raise RestoreError('보호 영상 해시 불일치: 원래 파일만 허용됩니다')
    validate_original(chunk,meta)
    for g,e in manifest['envelopes'].items():
        if len(unb64(e['nonce']))!=12 or len(e['wraps'])!=(1 if g=='master' else 2):
            raise RestoreError('nonce / 키 포장 오류')
        if any(len(unb64(w))!=40 for w in e['wraps']) or len(unb64(doc['seals'][g]))!=16:
            raise RestoreError('키 포장 / 인증 크기 오류')
        expected = sum(int(m[g].sum())*3 for m in masks) + 16
        actual=sections[g]['length'] if sections is not None else (chunk/(g+'.gcm')).stat().st_size
        if actual != expected:
            raise RestoreError('암호문 크기 오류')
        if digest(section_bytes(chunk,g,MAX_CHUNK_RAW+16)) != e['sha256']:
            raise RestoreError('암호문 해시 불일치')
    return doc, masks

def restore_frames(chunk, key=None):
    """No key discovery: accepts ONLY these provided 32 bytes. No pixel leaves on failure."""
    try:
        doc,masks=inspect_chunk(chunk)
        manifest=doc['manifest']; meta=manifest['meta']; decoded={}
        if key is not None:
            try:
                original,original_meta=restore_original_frames(chunk,key)
                return original,original_meta,list(GROUPS)
            except RestoreError as exc:
                if str(exc)!='전체키 인증 실패':
                    raise
        # Snapshot bytes and recheck digests; never decode a pathname that can change after authentication.
        video = section_bytes(chunk,'protected')
        if digest(video) != meta['protected_sha256']:
            raise RestoreError('보호 영상이 검증 중 변경되었습니다')
        ciphertexts = {}
        for g in GROUPS:
            ciphertexts[g] = section_bytes(chunk,g,MAX_CHUNK_RAW+16)
            if digest(ciphertexts[g]) != manifest['envelopes'][g]['sha256']:
                raise RestoreError('암호문이 검증 중 변경되었습니다')
        if key is not None:
            if len(key)!=32:
                raise RestoreError('키는 32바이트여야 합니다')
            for g,e in manifest['envelopes'].items():
                dek=None
                for wrapped in e['wraps']:
                    try:
                        dek=aes_key_unwrap(key,unb64(wrapped))
                        break
                    except InvalidUnwrap:
                        continue
                if dek is None:
                    continue
                AESGCM(dek).decrypt((1).to_bytes(12,'big'),unb64(doc['seals'][g]),canonical(manifest))
                raw=AESGCM(dek).decrypt(unb64(e['nonce']),ciphertexts[g],
                                       canonical({'meta':meta,'group':g}))
                expected=sum(int(m[g].sum())*3 for m in masks)
                if len(raw)!=expected:
                    raise RestoreError('픽셀 payload 길이 오류')
                decoded[g]=raw
            if not decoded:
                raise RestoreError('잘못된 키: 복원 권한 없음')
        fd = os.memfd_create('srx-protected', os.MFD_CLOEXEC)
        with os.fdopen(fd, 'w+b') as snapshot:
            snapshot.write(video)
            snapshot.flush()
            frames=read_video('/proc/self/fd/'+str(snapshot.fileno()),len(masks),meta['shape'])
        for g,raw in decoded.items():
            offset=0
            for frame,m in zip(frames,masks):
                length=int(m[g].sum())*3
                frame[m[g]]=np.frombuffer(raw[offset:offset+length],dtype=np.uint8).reshape(-1,3)
                offset+=length
        return frames,meta,sorted(decoded)
    except RestoreError:
        raise
    except Exception as exc:
        raise RestoreError('청크 인증 또는 형식 검증 실패') from exc

def restore_to(chunk, output, key=None):
    output=Path(output).absolute()
    if output.exists() or not any(output.resolve().is_relative_to(p) for p in (ROOT/'raw_data', ROOT/'tests/runtime')) or any(p.is_symlink() for p in (output,*output.parents)):
        raise RestoreError('복원 결과 경로 오류 또는 덮어쓰기 시도')
    # Authentication and all decoding finish before ANY restored output is created.
    frames,meta,groups=restore_frames(chunk,key)
    private_dir(output.parent)
    stage=private_dir(output.parent/('.restore-'+uuid.uuid4().hex))
    write_video(stage/'restored.avi',frames,meta['fps'])
    for i,f in enumerate(frames):
        ok,png=cv2.imencode('.png',f)
        if not ok:
            raise RestoreError('PNG 인코딩 실패')
        write_new(stage/f'{i:06d}.png',png.tobytes())
    write_new(stage/'result.json',canonical({'chunk':meta['chunk'],'groups':groups,'frames':len(frames),'fps':meta['fps']}))
    sync_dir(stage)
    os.rename(stage,output)
    sync_dir(output.parent)
    return {'output':str(output),'groups':groups,'frames':len(frames),'fps':meta['fps']}
