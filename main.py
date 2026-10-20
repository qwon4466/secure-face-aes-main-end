"""
SecureFace-RX v2 — 통합 FastAPI 서버

  GET  /                        → 실시간 모니터링 (SCR-002)
  GET  /register                → 사원 등록 UI
  GET  /assets                  → 보호 자산 목록 (SCR-003/004)
  GET  /employees               → 사원 관리 UI (SCR-006)
  GET  /stream/cam_0            → MJPEG 익명화 스트림
  GET  /api/stats               → 실시간 통계
  GET  /api/users               → 등록 사원 목록
  DELETE /api/users/{name}      → 사원 삭제 (SCR-006)
  POST /api/users/reload        → 카메라 DB 즉시 갱신
  POST /api/register            → 얼굴 등록 (3각도)
  GET  /api/assets              → 녹화 청크 목록
  GET  /api/assets/{chunk_id}   → 청크 상세
  GET  /decryption_page         → 원본 복원 (관리자)
  GET  /selective/              → 집단별 선택 복원 (관리자)

실행:
  python main.py
  또는
  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
_PROJECT_ROOT = Path(__file__).resolve().parent
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response, StreamingResponse)
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import UploadFile, File

import config as c
import event_log

from core.selective_camera import MainCamera as CameraProcessor, DB_PATH, _adapt_array, _convert_array
from core.selective_recognition import stable_person_id
from core.selective_routes import create_router as create_selective_router
from core.selective_index import completed as completed_selective, find as find_selective


IMAGE_DIR = str(_PROJECT_ROOT / "registered_faces")


_ACTIVE_DATA_ROOT = Path(os.environ.get('SRX_DATA_ROOT',str(c.RAW_DATA_DIR))).absolute()
_RAW_DATA_EXISTED = _ACTIVE_DATA_ROOT.is_dir()
camera = CameraProcessor(c)
if os.environ.get("SRX_TEST_MODE") in ("synthetic", "file", "camera"):
    DB_PATH = str(camera.root / "test-users.db")
    IMAGE_DIR = str(camera.root / "test-registered-faces")
c.RAW_DATA_DIR = camera.root
camera.gallery_path = DB_PATH

# AES 원본 복원은 전용 worker에서 실행한다. 단일 worker로 CPU 경쟁을
# 제한하면서도 FastAPI event loop와 HTTP 요청을 점유하지 않는다.
_RESTORE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aes-restore")
_RESTORE_LOCK = threading.Lock()
_RESTORE_JOBS: dict[str, dict] = {}


# ── DB 초기화 ─────────────────────────────────────────────────────────────
def _init_db():
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            auth_group TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vector     array NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

async def auto_recognize_logger():
    """60초마다 카메라 화면을 분석하여 터미널에 유사도를 자동 출력하는 백그라운드 작업"""
    print("\n🟢 [시스템] 60초 자동 분석 백그라운드 작업이 성공적으로 등록되었습니다!")
    
    from insightface.utils import face_align
    while True:
        await asyncio.sleep(60) # 60초 대기
        print("\n⏰ [시스템] 60초 주기 도달! 카메라 얼굴 분석을 시도합니다...")
        
        # 1. 예외 처리 (어디서 막히는지 확인)
        if camera is None:
            print("❌ [오류] camera 객체가 생성되지 않았습니다.")
            continue
        if camera.detector is None or camera.recognizer is None:
            print("❌ [오류] AI 모델(ArcFace)이 아직 로드되지 않았습니다.")
            continue
            
        frame = camera.capture_raw_frame()
        if frame is None:
            print("❌ [오류] 카메라에서 영상을 받아오지 못했습니다. (카메라 연결 확인 필요)")
            continue
            
        # 2. DB 확인
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        rows = conn.execute("SELECT name, vector FROM users").fetchall()
        conn.close()
        
        if not rows:
            print("⚠️ [결과] DB에 등록된 사원이 없어서 비교할 수 없습니다.")
            continue

        # 3. 얼굴 탐지
        bboxes, kpss = camera.detector.detect(frame, max_num=5, metric="default")
        if bboxes is None or len(bboxes) == 0:
            print("🔎 [결과] 현재 카메라 화면에 인식된 얼굴이 없습니다.")
            continue
            
        # 4. 분석 결과 출력
        print("\n" + "="*50)
        print("🕒 [60초 자동 분석] 실시간 카메라 얼굴 인식 결과")
        print("="*50)
        
        for i in range(len(bboxes)):
            try:
                lm = kpss[i]
                aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
                current_emb = camera.recognizer.get_feat(aligned)
                
                if current_emb is None:
                    print(f"▶ 탐지된 얼굴 {i+1}: ⚠️ 특징점 추출 실패")
                    continue
                    
                # 💡 핵심 수정 1: 연산 전 벡터를 1차원으로 확실하게 펴줌 (에러 방지)
                current_emb = np.array(current_emb).flatten()
                    
                best_match_name = "알 수 없음"
                best_score = 0.0
                
                for name, db_vector in rows:
                    # 💡 핵심 수정 2: DB에서 가져온 벡터도 1차원으로 펴줌
                    db_vector_flat = np.array(db_vector).flatten()
                    
                    # 0으로 나누는 에러 방지용 안전장치
                    norm_curr = np.linalg.norm(current_emb)
                    norm_db = np.linalg.norm(db_vector_flat)
                    
                    if norm_curr == 0 or norm_db == 0:
                        continue
                        
                    similarity = float(np.dot(current_emb, db_vector_flat) / (norm_curr * norm_db))
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match_name = stable_person_id(name)
                
                threshold = 0.45
                if best_score > threshold:
                    print(f"▶ 탐지된 얼굴 {i+1}: ✅ [등록 사원] {best_match_name} (유사도: {best_score:.4f})")
                else:
                    print(f"▶ 탐지된 얼굴 {i+1}: ❌ [비허가자] (최고 유사도: {best_score:.4f})")
                    
            except Exception as e:
                # 💡 에러가 나면 멈추지 않고 터미널에 원인을 출력
                print(f"▶ 탐지된 얼굴 {i+1} 분석 중 에러 발생: {e}")
                
        print("="*50 + "\n")
        
# ── Lifespan (FastAPI 0.93+ 권장 방식) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    c.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = c.RAW_DATA_DIR / ('.write-check-' + uuid.uuid4().hex)
    try:
        with probe.open('xb') as out:
            out.write(b'1');out.flush();os.fsync(out.fileno())
    except OSError as exc:
        raise RuntimeError(f'[STORAGE][ERROR] raw_data 폴더에 쓰기 권한이 없습니다 | 경로={c.RAW_DATA_DIR}') from exc
    finally:
        probe.unlink(missing_ok=True)
    if _RAW_DATA_EXISTED:
        print(f'[STORAGE] 기존 raw_data 폴더 사용 | 경로={c.RAW_DATA_DIR} | 완료청크={len(completed_selective(c.RAW_DATA_DIR))}개',flush=True)
    else:
        print(f'[STORAGE] raw_data 폴더 생성 완료 | 경로={c.RAW_DATA_DIR}',flush=True)
    
    # 1. 시스템 DB 초기화
    os.makedirs(IMAGE_DIR, exist_ok=True)
    _init_db()
    
    # 2. CPU 선택 복원 파이프라인 시작 (전체 원본 녹화기 없음)
    camera.start()

    # 👇 여기에 이 한 줄을 추가하세요! (30초 타이머 시작) 👇
    # Recognition is performed once by the selective camera pipeline.

    yield  # --- 서버 가동 중 ---
    


    # 서버 종료 시 안전한 자원 해제
    if camera:
        camera.stop()


# ── 앱 초기화 ─────────────────────────────────────────────────────────────
app = FastAPI(title="SecureFace-RX v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=str(_PROJECT_ROOT / "templates"))


@app.middleware("http")
async def selective_response_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/selective"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ── 관리자 인증 (사원 등록/관리 접근 제어) ────────────────────────────────
# 서버 측 세션(쿠키) 기반. 로그인 성공 시 토큰을 발급해 쿠키로 저장하고,
# 보호 대상 페이지·API는 이 쿠키를 검증한다. 토큰은 메모리 보관(서버 재시작 시 초기화).
_ADMIN_SESSIONS: dict = {}   # token -> admin_id
_ADMIN_COOKIE = "admin_session"
_ADMIN_STORE = "admin_users.json"   # 회원가입으로 추가된 관리자 계정 저장


def _is_admin(request: Request) -> bool:
    tok = request.cookies.get(_ADMIN_COOKIE)
    return bool(tok) and tok in _ADMIN_SESSIONS


def _admin_id(request: Request) -> str:
    tok = request.cookies.get(_ADMIN_COOKIE, "")
    return _ADMIN_SESSIONS.get(tok, "")


def require_admin(request: Request):
    """보호 API용 의존성. 미로그인 시 401."""
    if not _is_admin(request):
        raise HTTPException(status_code=401, detail="관리자 로그인이 필요합니다.")


def _audit_selective(request, chunk_id, success, role):
    try:
        chunk_name=find_selective(camera.chunks,chunk_id).relative_to(camera.chunks).as_posix()
    except Exception:
        chunk_name=chunk_id
    event_log.log_restore(chunk_name, _admin_id(request), request.client.host if request.client else '-', success,
                          mode='selective-'+role)

app.include_router(create_selective_router(camera, require_admin, _audit_selective))


def _hash_pw(pw: str, salt: str) -> str:
    return hashlib.sha256((salt + pw).encode("utf-8")).hexdigest()


def _load_admins() -> dict:
    try:
        with open(_ADMIN_STORE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_admins(d: dict):
    with open(_ADMIN_STORE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def _check_admin(admin_id: str, pw: str) -> bool:
    # 1) config의 기본 관리자 계정
    if admin_id == getattr(c, "ADMIN_ID", "") and pw == getattr(c, "ADMIN_PASSWORD", ""):
        return True
    # 2) 회원가입으로 등록된 계정
    rec = _load_admins().get(admin_id)
    return bool(rec) and _hash_pw(pw, rec.get("salt", "")) == rec.get("hash")


class AdminLogin(BaseModel):
    id: str = ""
    password: str


@app.get("/admin/login", response_class=HTMLResponse)
async def page_admin_login(request: Request):
    # 이미 로그인돼 있으면 사원 관리로
    if _is_admin(request):
        return RedirectResponse("/employees", status_code=302)
    return templates.TemplateResponse(request=request, name="admin_login.html")


@app.post("/api/admin/login")
async def api_admin_login(data: AdminLogin):
    if not _check_admin(data.id, data.password):
        raise HTTPException(status_code=401,
                            detail="아이디 또는 비밀번호가 일치하지 않습니다.")
    tok = secrets.token_urlsafe(24)
    _ADMIN_SESSIONS[tok] = data.id
    resp = JSONResponse({"ok": True})
    resp.set_cookie(_ADMIN_COOKIE, tok, httponly=True,
                    max_age=8 * 3600, samesite="lax")
    return resp


@app.post("/api/admin/signup")
async def api_admin_signup(data: AdminLogin):
    """관리자 회원가입 — 새 관리자 계정 생성."""
    if not getattr(c, "ADMIN_ALLOW_SIGNUP", False):
        raise HTTPException(status_code=403,
                            detail="회원가입이 비활성화되어 있습니다. 관리자에게 문의하세요.")
    admin_id = (data.id or "").strip()
    if not admin_id or not data.password:
        raise HTTPException(status_code=400, detail="아이디와 비밀번호를 입력하세요.")
    if len(data.password) < 4:
        raise HTTPException(status_code=400, detail="비밀번호는 4자 이상이어야 합니다.")
    admins = _load_admins()
    if admin_id == getattr(c, "ADMIN_ID", "") or admin_id in admins:
        raise HTTPException(status_code=409, detail="이미 존재하는 아이디입니다.")
    salt = secrets.token_hex(8)
    admins[admin_id] = {"salt": salt, "hash": _hash_pw(data.password, salt)}
    _save_admins(admins)
    return {"ok": True}


@app.post("/api/admin/logout")
async def api_admin_logout(request: Request):
    tok = request.cookies.get(_ADMIN_COOKIE)
    if tok:
        _ADMIN_SESSIONS.pop(tok, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_ADMIN_COOKIE)
    return resp


@app.get("/api/admin/status")
async def api_admin_status(request: Request):
    return {
        "logged_in": _is_admin(request),
        "who": _admin_id(request),
        "allow_signup": bool(getattr(c, "ADMIN_ALLOW_SIGNUP", False)),
    }


# ── 페이지 ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_monitor(request: Request):
    return templates.TemplateResponse(request=request, name="monitor.html")


@app.get("/register", response_class=HTMLResponse)
async def page_register(request: Request):
    # 관리자 로그인 필요 → 미로그인 시 로그인 페이지로
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/register", status_code=302)
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/assets", response_class=HTMLResponse)
async def page_assets(request: Request):
    return templates.TemplateResponse(request=request, name="assets.html")


@app.get("/employees", response_class=HTMLResponse)
async def page_employees(request: Request):
    # 관리자 로그인 필요 → 미로그인 시 로그인 페이지로
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/employees", status_code=302)
    return templates.TemplateResponse(request=request, name="employees.html")


# ── 스냅샷 (단일 JPEG, JS 폴링용) ────────────────────────────────────────
@app.get("/snapshot/cam_0")
async def snapshot_cam0():
    jpeg = camera.get_jpeg() if camera else None
    if jpeg is None:
        # 카메라 준비 중 — 빈 1×1 회색 JPEG 반환
        import base64
        placeholder = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
            "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
            "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
            "/9oADAMBAAIRAxEAPwCwABmX/9k="
        )
        return Response(content=placeholder, media_type="image/jpeg")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )

# ── 등록용 원본 스냅샷 (익명화 전) ──────────────────────────────────────
@app.get("/snapshot/raw", dependencies=[Depends(require_admin)])
async def snapshot_raw():
    jpeg = camera.get_raw_jpeg() if camera else None
    if jpeg is None:
        raise HTTPException(status_code=503, detail="카메라 준비 중")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


# ── MJPEG 스트리밍 (호환 브라우저용, 유지) ───────────────────────────────
@app.get("/stream/cam_0")
async def stream_cam0():
    async def generate():
        while True:
            jpeg = camera.get_jpeg() if camera else None
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── API — 통계 ────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def api_stats():
    s = camera.get_stats() if camera else {"employee_count": 0, "unknown_count": 0}
    s["unknown_alerts"] = s.get("unknown_count", 0)  # 현재 프레임 기준; 누적 중복 없음
    for field in ("chunks", "session"):
        s.pop(field, None)
    return s


# ── API — 디버그 (카메라 상태 확인) ──────────────────────────────────────
@app.get("/api/debug", dependencies=[Depends(require_admin)])
async def api_debug():
    if camera is None:
        return {"error": "camera not initialized"}
    return camera.get_debug_info()


# ── API — 사원 목록 ───────────────────────────────────────────────────────
@app.get("/api/users", dependencies=[Depends(require_admin)])
async def api_users():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name, auth_group FROM users").fetchall()
    conn.close()
    seen = {}
    for stored_name, group in rows:
        seen[stable_person_id(stored_name)] = group
    return [{"name": k, "group": v} for k, v in seen.items()]


# ── API — DB 재로드 ───────────────────────────────────────────────────────
@app.post("/api/users/reload", dependencies=[Depends(require_admin)])
async def api_reload():
    if camera:
        camera.reload_db()
    return {"status": "ok"}


# ── API — 사원 삭제 (Phase 6) ─────────────────────────────────────────────
@app.delete("/api/users/{name}", dependencies=[Depends(require_admin)])
async def api_delete_user(name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "DELETE FROM users WHERE name LIKE ?", (f"{name}_%",)
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="사원을 찾을 수 없습니다.")
    if camera:
        camera.reload_db()
    return {"status": "ok", "deleted": deleted}


# ── API — 얼굴 등록 ───────────────────────────────────────────────────────
class RegisterData(BaseModel):
    name: str
    group: str
    image_base64: str


class RegisterCapture(BaseModel):
    name: str
    group: str


def _do_register(frame, name: str, group: str) -> dict:
    """단일 프레임으로 얼굴 탐지 → 각도 판별 → DB 등록."""
    if frame is None:
        return {"status": "error", "message": "❌ 카메라 프레임이 없습니다."}

    from insightface.utils import face_align
    detector = camera.detector
    recognizer = camera.recognizer

    bboxes, kpss = detector.detect(frame, max_num=1, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return {"status": "error", "message": "❌ 얼굴을 찾을 수 없습니다."}

    x1, y1, x2, y2 = bboxes[0, :4].astype(int).tolist()
    lm = kpss[0]

    # 측면 판별 (눈-코 거리 비율)
    le, re, nose = lm[0], lm[1], lm[2]
    d_left = np.linalg.norm(le - nose)
    d_right = np.linalg.norm(re - nose)
    ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
    eye_dist = np.linalg.norm(le - re)
    is_side = ratio > 1.5 or (eye_dist / (x2 - x1 + 1e-5)) < 0.25

    aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
    emb = recognizer.get_feat(aligned)
    if emb is None:
        return {"status": "error", "message": "❌ 임베딩 추출 실패"}

    if not is_side:
        tag, fname = "정면", f"{name}_정면.jpg"
    else:
        if d_left < d_right:
            tag, fname = "좌측면", f"{name}_측면1(좌).jpg"
        else:
            tag, fname = "우측면", f"{name}_측면2(우).jpg"

    fpath = os.path.join(IMAGE_DIR, fname)
    cv2.imwrite(fpath, frame)

    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute(
        "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)",
        (f"{name}_{tag}", group, fpath, emb),
    )
    conn.commit()
    conn.close()

    camera.reload_db()
    return {"status": "success", "message": f"✅ [{name}] {tag} 등록 성공!"}


@app.post("/api/register", dependencies=[Depends(require_admin)])
async def api_register(data: RegisterData):
    """브라우저가 캡처한 base64 이미지로 등록 (기존 호환)."""
    try:
        if camera is None:
            return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}
        _, encoded = data.image_base64.split(",", 1)
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR,
        )
        return _do_register(frame, data.name, data.group)
    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {e}"}


@app.post("/api/register_capture", dependencies=[Depends(require_admin)])
async def api_register_capture(data: RegisterCapture):
    """서버(RealSense) 카메라의 현재 원본 프레임으로 등록."""
    try:
        if camera is None:
            return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}
        frame = camera.capture_raw_frame()
        return _do_register(frame, data.name, data.group)
    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {e}"}

# ── API — 연속 촬영 얼굴 등록 (애플 Face ID 스타일) ──────────────────────
class RegisterContinuous(BaseModel):
    name: str
    group: str

@app.post("/api/register_continuous_realsense", dependencies=[Depends(require_admin)])
async def api_register_continuous_realsense(data: RegisterContinuous):
    """
    프론트엔드의 트리거 신호를 받아 약 3~4초간 연속으로 프레임을 캡처하고,
    얼굴을 탐지하여 다각도의 특징점(Vector)을 DB에 일괄 등록합니다.
    """
    if camera is None:
        return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}

    print(f"▶ [{data.name}] 연속 촬영 등록 시작...")
    
    capture_duration = 5.0  # 5초간 캡처
    interval = 0.2          # 0.2초 간격 (초당 5프레임)
    max_frames = int(capture_duration / interval)
    
    valid_faces = 0
    from insightface.utils import face_align
    detector = camera.detector
    recognizer = camera.recognizer
    
    # DB 연결
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    
    try:
        for i in range(max_frames):
            # 1. RealSense 카메라 원본 프레임 가져오기
            frame = camera.capture_raw_frame()
            if frame is None:
                await asyncio.sleep(interval)
                continue
            
            # 2. 얼굴 탐지 (SCRFD)
            bboxes, kpss = detector.detect(frame, max_num=1, metric="default")
            if bboxes is not None and len(bboxes) > 0:
                lm = kpss[0]
                
                # 3. 얼굴 정렬 및 임베딩(특징점) 추출
                aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
                emb = recognizer.get_feat(aligned)
                
                if emb is not None:
                    # 파일명 생성 및 이미지 저장
                    fname = f"{data.name}_연속_{i}.jpg"
                    fpath = os.path.join(IMAGE_DIR, fname)
                    cv2.imwrite(fpath, frame)
                    
                    # DB 저장 (이름 뒤에 _연속_번호 를 붙여 구분)
                    conn.execute(
                        "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)",
                        (f"{data.name}_연속_{i}", data.group, fpath, emb),
                    )
                    valid_faces += 1
            
            # 다음 캡처까지 대기 (비동기 딜레이)
            await asyncio.sleep(interval)
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ 등록 중 오류 발생: {e}")
        return {"status": "error", "message": f"등록 중 서버 오류 발생: {e}"}
    finally:
        conn.close()

    # 4. 결과 판별 및 DB 리로드
    if valid_faces > 0:
        camera.reload_db()  # 새로 등록된 데이터 즉시 반영
        print(f"✅ [{data.name}] 총 {valid_faces}개의 얼굴 각도 등록 완료.")
        return {"status": "success", "message": f"총 {valid_faces}개의 얼굴 각도가 성공적으로 등록되었습니다."}
    else:
        return {"status": "error", "message": "얼굴을 제대로 인식하지 못했습니다. 밝은 곳에서 정면을 보고 다시 시도해주세요."}

# ── API — 보호 자산 목록 (Phase 4) ───────────────────────────────────────
@app.get("/api/assets", dependencies=[Depends(require_admin)])
async def api_assets():
    return camera.list_assets() + _list_raw_asset_chunks()


def _raw_asset_id(relative_dir: str) -> str:
    """raw_data 상대 경로를 기존 assets API의 안전한 chunk_id로 변환한다."""
    return relative_dir.replace(os.sep, "__").replace("/", "__")


def _list_raw_asset_chunks() -> list[dict]:
    """실제 AES raw 저장 구조를 assets 화면용 레코드로 변환한다."""
    base = os.path.realpath(c.RAW_DATA_DIR)
    if not os.path.isdir(base):
        return []
    records = []
    for root, _dirs, files in os.walk(base):
        if 'manifest.json' in files or any(part.startswith('.') for part in Path(root).relative_to(base).parts):
            continue
        for name in files:
            if not name.endswith(".enc"):
                continue
            stem = name[:-4]
            meta_path = os.path.join(root, stem + ".json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            rel_dir = os.path.relpath(root, base)
            parts = rel_dir.split(os.sep)
            if len(parts) < 2:
                continue
            date_name = parts[0]
            start_time = meta.get("chunk_id") or (parts[-1] if parts else None)
            if not start_time or start_time.startswith("chunk_"):
                start_time = meta.get("start_time", "") or (parts[-1] if parts else "")
            if start_time.count("-") == 2 and ":" not in start_time:
                start_time = start_time.replace("-", ":")
            fps = float(meta.get("fps", 0) or 0)
            frame_count = int(meta.get("frame_count", 0) or 0)
            duration = (frame_count / fps) if fps > 0 else float(meta.get("chunk_seconds", 0) or 0)
            records.append({
                "chunk_id": _raw_asset_id(rel_dir),
                "date": date_name,
                "start_time": start_time,
                "duration_seconds": round(duration, 2),
                "frame_count": frame_count,
                "total_faces": 0,
                # 구형 단일 encrypted_raw 파일은 complete 필드가 없으므로
                # 암호화 파일과 메타데이터가 함께 있으면 완료된 자산으로 본다.
                "complete": bool(meta.get("complete", True)),
                "last_update": meta.get("updated_at", ""),
                "format": "raw",
            })
    records.sort(key=lambda item: (item.get("date", ""), item.get("start_time", "")), reverse=True)
    return records


@app.get("/api/save_progress", dependencies=[Depends(require_admin)])
async def api_save_progress():
    """진행 중인 청크의 저장 진행률·ETA (자산 화면 상단 배너용)."""
    return camera.get_save_progress()


# ── API — 감사/이벤트 로그 + 구형 상태 호환 API ─────────────────────────────
@app.get("/api/audit/restore", dependencies=[Depends(require_admin)])
async def api_audit_restore():
    """복원(익명화 해제) 접근 이력."""
    names={}
    for manifest in c.RAW_DATA_DIR.glob('*/*/*/manifest.json'):
        try:
            meta=json.loads(manifest.read_text())['manifest']['meta']
            names[meta['chunk']]=manifest.parent.relative_to(c.RAW_DATA_DIR).as_posix()
        except (OSError,ValueError,KeyError):
            continue
    return [entry | {'chunk_name':names.get(entry.get('chunk_id'),
                     entry.get('chunk_id','') if '/' in entry.get('chunk_id','') else '경로 정보 없음')}
            for entry in event_log.read_restore(300)]


@app.get("/api/audit/detections", dependencies=[Depends(require_admin)])
async def api_audit_detections():
    """비허가자 감지 이벤트 이력."""
    return event_log.read_detections(300)


@app.get("/audit", response_class=HTMLResponse)
async def page_audit(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/audit", status_code=302)
    return templates.TemplateResponse(request=request, name="audit.html")


@app.get("/api/assets/{chunk_id}", dependencies=[Depends(require_admin)])
async def api_asset_detail(chunk_id: str):
    for item in camera.list_assets() + _list_raw_asset_chunks():
        if item["chunk_id"] == chunk_id:
            return item | {"frames": []}
    raise HTTPException(status_code=404, detail="청크를 찾을 수 없습니다.")


@app.get("/decryption_page")
async def decryption_page(request: Request):
    """상단 메뉴에서 '원본 복호화' 버튼 클릭 시 열리는 전용 웹페이지"""
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/decryption_page", status_code=302)
    return templates.TemplateResponse(request=request, name="decryption.html", context={"request": request})

@app.get("/api/admin/chunks", dependencies=[Depends(require_admin)])
async def get_chunk_list():
    """raw_data/YYYY-MM-DD/HH/HH:MM:SS 구조와 구형 구조를 함께 탐색한다."""
    base_dir = str(c.RAW_DATA_DIR)
    chunks = {}
    for path,meta in completed_selective(c.RAW_DATA_DIR):
        relative=path.relative_to(c.RAW_DATA_DIR)
        chunks.setdefault(relative.parts[0],[]).append('/'.join(relative.parts[1:]))
    
    if not os.path.exists(base_dir):
        return JSONResponse(content={})
        
    for date_folder in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
        date_chunks = []
        for level1_folder in sorted(os.listdir(date_path), reverse=True):
            level1_path = os.path.join(date_path, level1_folder)
            if not os.path.isdir(level1_path):
                continue

            # 신규: 날짜/HH/HH:MM:SS/HH:MM:SS.enc
            if level1_folder.isdigit() and len(level1_folder) == 2:
                for chunk_name in sorted(os.listdir(level1_path), reverse=True):
                    chunk_path = os.path.join(level1_path, chunk_name)
                    enc_path = os.path.join(chunk_path, f"{chunk_name}.enc")
                    meta_path = os.path.join(chunk_path, f"{chunk_name}.json")
                    if not (os.path.isdir(chunk_path) and
                            os.path.isfile(enc_path) and os.path.isfile(meta_path)):
                        continue
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except (OSError, ValueError):
                        continue
                    if meta.get("complete", False):
                        date_chunks.append(f"{level1_folder}/{chunk_name}")

            # 구형: 날짜/시간/chunk_XXXX 또는 날짜/시간/encrypted_raw.bin.enc
            for chunk_name in sorted(os.listdir(level1_path), reverse=True):
                chunk_path = os.path.join(level1_path, chunk_name)
                meta_path = os.path.join(chunk_path, f"{chunk_name}.json")
                enc_path = os.path.join(chunk_path, f"{chunk_name}.enc")
                if (chunk_name.startswith("chunk_") and
                        os.path.isdir(chunk_path) and
                        os.path.isfile(enc_path) and os.path.isfile(meta_path)):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except (OSError, ValueError):
                        continue
                    if meta.get("complete", False):
                        date_chunks.append(f"{level1_folder}/{chunk_name}")

            if (os.path.isfile(os.path.join(level1_path, "encrypted_raw.bin.enc"))
                    and os.path.isfile(os.path.join(level1_path, "encrypted_raw.bin.json"))):
                date_chunks.append(level1_folder)
        if date_chunks:
            chunks[date_folder] = date_chunks
                    
    return JSONResponse(content=chunks)

@app.post("/api/admin/verify_chunk", dependencies=[Depends(require_admin)])
async def verify_chunk(request: Request, folder_path: str = Form(...), password: str = Form(...)):
    """복원 job을 등록하고 즉시 반환한다."""
    enc_file, json_file = _resolve_restore_files(folder_path)
    if os.path.isdir(enc_file) and not _check_admin(_admin_id(request), password):
        raise HTTPException(status_code=403, detail="관리자 비밀번호가 일치하지 않습니다.")
    job_id = uuid.uuid4().hex[:12]
    with _RESTORE_LOCK:
        _RESTORE_JOBS[job_id] = {
            "status": "queued", "progress": 0,
            "message": "복원 대기 중", "folder_path": folder_path,
        }
    _RESTORE_EXECUTOR.submit(_run_restore_job, job_id, enc_file, json_file, password)
    print(f"[RESTORE] 작업 등록 | Job ID: {job_id} | 대상: {folder_path}")
    return {"job_id": job_id, "status": "queued"}


def _resolve_restore_files(folder_path: str) -> tuple[str, str]:
    """raw_data 아래의 암호화 청크만 복원 대상으로 허용한다."""
    base = os.path.realpath(c.RAW_DATA_DIR)
    folder = os.path.realpath(folder_path)
    if folder != base and not folder.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="허용되지 않은 청크 경로입니다.")
    if (os.path.isfile(os.path.join(folder, "manifest.json"))
            or os.path.isfile(os.path.join(folder, "encrypted_raw.bin.json"))):
        try:
            from core.selective_crypto import inspect_chunk
            inspect_chunk(folder)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="완료된 원본 청크를 검증할 수 없습니다.") from exc
        return folder, os.path.join(folder, "encrypted_raw.bin.json")
    # 신규 포맷: raw_data/.../HH/HH:MM:SS/HH:MM:SS.enc + .json
    # 구형 포맷: raw_data/.../chunk_0001/chunk_0001.enc + .json
    chunk_encs = sorted(
        name for name in os.listdir(folder)
        if name.endswith(".enc") and name != "encrypted_raw.bin.enc"
    ) if os.path.isdir(folder) else []
    is_chunk = bool(chunk_encs)
    stem = ""
    if is_chunk:
        enc_name = chunk_encs[0]
        stem = enc_name[:-4]
        enc_file = os.path.join(folder, enc_name)
        json_file = os.path.join(folder, stem + ".json")
    else:
        # 구 포맷: raw_data/date/time/encrypted_raw.bin.enc
        enc_file = os.path.join(folder, "encrypted_raw.bin.enc")
        json_file = os.path.join(folder, "encrypted_raw.bin.json")
    if any(not os.path.realpath(p).startswith(base + os.sep) for p in (enc_file, json_file)):
        raise HTTPException(status_code=400, detail="허용되지 않은 파일 경로입니다.")
    if not os.path.isfile(enc_file) or not os.path.isfile(json_file):
        raise HTTPException(status_code=404, detail="암호화된 파일이나 메타데이터를 찾을 수 없습니다.")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if is_chunk and not meta.get("complete", False):
            raise HTTPException(status_code=409, detail="아직 완료되지 않은 청크입니다.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"청크 메타데이터를 읽을 수 없습니다: {exc}")
    return enc_file, json_file


def _set_restore_job(job_id: str, **values):
    with _RESTORE_LOCK:
        job = _RESTORE_JOBS.get(job_id)
        if job is not None:
            job.update(values)


def _run_restore_job(job_id: str, enc_file: str, json_file: str, password: str):
    """현재 AES-GCM 또는 과거 AES 청크를 복호화해 MP4로 조립한다."""
    from utils.aes_crypto import iter_decrypted_frames
    import shutil
    import subprocess

    started = time.perf_counter()
    selective = os.path.isdir(enc_file)
    result_dir = c.RAW_DATA_DIR / ".results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_base = str(result_dir / ("original-" + job_id)) if selective else enc_file[:-4] + f".{job_id}"
    temp_video = result_base + ".tmp.avi"
    temp_mp4 = result_base + ".tmp.mp4"
    final_mp4 = result_base + ".mp4"
    writer = None
    ffmpeg_proc = None
    try:
        print("[RESTORE] ========================================")
        print(f"[RESTORE] 복원 작업 시작 | Job ID: {job_id}")
        print(f"[RESTORE] 대상: {enc_file if selective else os.path.dirname(enc_file)}")
        print("[RESTORE] 1. 청크 검색 시작 | 청크 개수: 1")
        _set_restore_job(job_id, status="processing", progress=5, message="AES 복호화 중")
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        restored_frames = None
        if selective:
            from core.selective_crypto import restore_original_frames
            master_key = (camera.keydir / "master.key").read_bytes()
            restored_frames, chunk_meta = restore_original_frames(enc_file, master_key)
            height, width, channels = chunk_meta["shape"]
            fps = float(chunk_meta["fps"])
        else:
            width = int(meta.get("width", 640))
            height = int(meta.get("height", 480))
            channels = int(meta.get("channels", 3))
            fps = float(meta.get("fps", 10))
        frame_size = width * height * channels
        if channels != 3 or frame_size <= 0:
            raise ValueError("지원하지 않는 프레임 형식입니다.")
        if selective:
            frame_count = len(restored_frames)
            total_bytes = frame_count * frame_size
        else:
            with open(enc_file, "rb") as f:
                size_raw = f.read(8)
            total_bytes = int.from_bytes(size_raw, "big") if len(size_raw) == 8 else 0
            frame_count = total_bytes // frame_size if frame_size else 0
        if total_bytes == 0 or total_bytes % frame_size:
            raise ValueError("복호화 데이터 크기가 프레임 크기와 일치하지 않습니다.")

        print("[RESTORE] 2. AES 복호화 시작")
        _set_restore_job(job_id, progress=20, message="영상 조립 중")
        print("[RESTORE] 3. 영상 조립 시작")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            ffmpeg_proc = subprocess.Popen(
                [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", temp_mp4],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            writer = cv2.VideoWriter(
                temp_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("영상 출력 파일을 열 수 없습니다.")

        decrypt_seconds = 0.0
        assemble_seconds = 0.0
        frames = iter(restored_frames) if selective else iter_decrypted_frames(enc_file, frame_size, password)
        output_started = time.perf_counter()
        for index in range(frame_count):
            t_dec = time.perf_counter()
            raw_data = next(frames, None)
            decrypt_seconds += time.perf_counter() - t_dec
            if raw_data is None:
                raise ValueError(f"프레임 {index + 1} 데이터가 손상되었습니다.")
            img = raw_data if selective else np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (height, width, channels))
            t_asm = time.perf_counter()
            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(img.tobytes())
            else:
                writer.write(img)
            assemble_seconds += time.perf_counter() - t_asm
            if frame_count >= 20 and (index + 1) % max(1, frame_count // 10) == 0:
                progress = 20 + int((index + 1) / frame_count * 55)
                _set_restore_job(
                    job_id, progress=progress,
                    message=f"영상 조립 중 ({index + 1}/{frame_count})",
                )
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=1800)
            if ffmpeg_proc.returncode != 0:
                raise RuntimeError("FFmpeg 영상 생성에 실패했습니다.")
            ffmpeg_proc = None
        else:
            writer.release()
            writer = None
        print(f"[RESTORE] 영상 조립 완료 | {assemble_seconds:.2f}s | {frame_count} frames")

        _set_restore_job(job_id, progress=80, message="최종 파일 생성 중")
        print("[RESTORE] 4. 최종 파일 생성")
        os.replace(temp_mp4 if ffmpeg else temp_video, final_mp4)
        file_seconds = time.perf_counter() - output_started
        total_seconds = time.perf_counter() - started
        print(f"[PERF] AES decrypt : {decrypt_seconds:.2f}s")
        print(f"[PERF] throughput  : {total_bytes / 1048576 / max(decrypt_seconds, 1e-9):.1f} MB/s")
        print(f"[PERF] assemble    : {assemble_seconds:.2f}s")
        print(f"[PERF] file write  : {file_seconds:.2f}s")
        print(f"[PERF] TOTAL       : {total_seconds:.2f}s")
        print(f"[RESTORE] 복원 완료 | 결과 파일: {final_mp4}")
        print("[RESTORE] ========================================")
        _set_restore_job(job_id, status="completed", progress=100,
                         message="복원 완료", file=os.path.basename(final_mp4), result_path=final_mp4)
    except Exception as exc:
        print(f"[RESTORE][ERROR] Job ID {job_id}: {exc}")
        _set_restore_job(job_id, status="failed", progress=0, message=str(exc), error=str(exc))
    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait(timeout=5)
            except Exception:
                pass
        if writer is not None:
            writer.release()
        for path in (temp_video, temp_mp4):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"[RESTORE][ERROR] 임시 파일 삭제 실패: {path}: {exc}")


@app.get("/api/admin/restore/status/{job_id}", dependencies=[Depends(require_admin)])
async def restore_status(job_id: str):
    with _RESTORE_LOCK:
        job = _RESTORE_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="복원 작업을 찾을 수 없습니다.")
        return {k: v for k, v in job.items() if k not in ("folder_path", "result_path")}


@app.get("/api/admin/restore/result/{job_id}", dependencies=[Depends(require_admin)])
async def restore_result(job_id: str):
    with _RESTORE_LOCK:
        job = _RESTORE_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="복원 작업을 찾을 수 없습니다.")
        if job.get("status") != "completed":
            raise HTTPException(status_code=409, detail="복원이 아직 완료되지 않았습니다.")
        filename = job.get("file", "")
    with _RESTORE_LOCK:
        result_path = _RESTORE_JOBS[job_id].get("result_path", "")
    if not os.path.isfile(result_path):
        raise HTTPException(status_code=404, detail="복원 결과 파일을 찾을 수 없습니다.")
    return FileResponse(result_path, media_type="video/mp4", filename="restored_raw_video.mp4")

# ── 직접 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
# 1.python main.py 실행
# 2. 터미널 하나 더 열고,  ngrok http --domain=appointee-dreary-unisexual.ngrok-free.dev 8000 입력
# 3. 주소창에 https://appointee-dreary-unisexual.ngrok-free.dev 입력
