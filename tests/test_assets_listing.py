import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def raw_asset_lister(data_dir):
    """main.py 전체(카메라 초기화)를 띄우지 않고 과거 형식 목록 함수만 꺼내 실행한다."""
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in ('_raw_asset_id', '_list_raw_asset_chunks')]
    ns = {'os': os, 'json': json, 'Path': Path, 'c': SimpleNamespace(RAW_DATA_DIR=data_dir)}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'main.py', 'exec'), ns)
    return ns['_list_raw_asset_chunks']


def test_current_container_chunks_are_not_listed_as_legacy_raw(work):
    current = work / '2026-10-21' / '15' / '15-20-00'
    current.mkdir(parents=True)
    (current / 'encrypted_raw.bin.enc').write_bytes(b'x')
    (current / 'encrypted_raw.bin.json').write_text(json.dumps({'format': 'srx-container-v2', 'status': 'completed'}))
    legacy = work / '2026-09-15' / '00' / 'chunk_0001'
    legacy.mkdir(parents=True)
    (legacy / 'chunk_0001.enc').write_bytes(b'x')
    (legacy / 'chunk_0001.json').write_text(json.dumps({'fps': 10, 'frame_count': 4}))

    records = raw_asset_lister(work)()
    # 현재 형식은 camera.list_assets()가 'selective'로 보여 주므로 여기서는 빠져야 한다.
    # 같이 나오면 화면 트리의 같은 자리에서 진짜 항목을 덮어써 "과거 저장 형식"으로 보인다.
    assert [(r['chunk_id'], r['format']) for r in records] == [('2026-09-15__00__chunk_0001', 'raw')]
