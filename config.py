"""
SecureFace-RX 전역 설정
(123.txt의 설정을 기준으로 zxc.txt의 디바이스, 아키텍처, 난독화 등 누락된 내용 병합)
"""

import os
import platform
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
# ─── 레거시 학습 설정(실행 경로에서 사용하지 않음) ───────────────────────
INV_BLOCKS   = 3       # INV_block_affine 반복 수 (config.INV_BLOCKS)
channels_in  = 3       # 입력 채널 수 (RGB)
clamp        = 2.0     # affine 스케일 클램핑 계수

# ─── 오복원(Wrong Recovery) 모드 (zxc.txt 기준) ───────────────────
# 'Random': RandWR — 랜덤 노이즈형 오복원 (PSNR<11dB)
# 'Obfs'  : ObfsWR — 난독화 유지형 오복원
WRONG_RECOVER_TYPE = 'Random'

# ─── 키 보조입력 정책 (zxc.txt 기준) ──────────────────────────────
SECRET_KEY_AS_NOISE = True  
# 복원 보조입력으로 K를 3채널 반복

# ─── Utility 조건부 기능 (기본 비활성) (zxc.txt 기준) ──────────────
ADJ_UTILITY = False

# ─── 정규화 해상도 (zxc.txt 기준) ─────────────────────────────────
# 원본 config: cropsize=224, SRS: NORM_RESOLUTION=256
# 공식 가중치 사용 시 학습된 해상도에 맞춰야 함
NORM_RESOLUTION = 256   # 변경 시 key 길이도 달라짐

# ─── 사전 난독화 (zxc.txt 기준) ───────────────────────────────────
DEFAULT_OBFUSCATOR = 'blur'
BLUR_KERNEL_SIZE   = 61
BLUR_SIGMA         = 21.0     # 원본 hybridAll: Blur(61, 9, 21)
BLUR_SIGMA_MIN     = 9.0      # 원본 hybridAll blur sigma_min
PIXELATE_BLOCK     = 20       
# 원본 hybridAll: Pixelate(20)
MEDIAN_KERNEL      = 23       # 원본 hybridAll: MedianBlur(23)

# ─── 검출기 (zxc.txt 기준) ────────────────────────────────────────
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_NMS_IOU        = 0.4
FACE_MARGIN             = 0.10


# ─── 학습 하이퍼파라미터 (123.txt 기준) ───────────────────────────
lr           = 0.00001
batch_size   = 6
weight_decay = 1e-5
init_scale   = 0.01
TRIPLET_MARGIN         = 1.2
LAMBDA_RECONSTRUCTION  = 5
LAMBDA_GUIDE           = 1
LAMBDA_LOW_FREQUENCY   = 1

SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

# ─── KeyGen (PBKDF2) ── NFR-SEC-2 경고 ───────────────────────────
# !!
# salt=1, count=10 은 논문의 "demonstration only" 값 !!
# 운영 배포 시 임의 salt + OWASP 권고 반복 수(≥600000)로 교체할 것
KEY_SALT  = 1
KEY_COUNT = 10

# ─── 기타 ─────────────────────────────────────────────────────────
debug = False
recognizer = 'AdaFaceIR100'

# ─── 통합 시스템 설정 (123.txt 우선) ───────────────────────────────
# 복원 비밀번호 (데모용 — 실제 배포 시 환경변수로 교체)
DEMO_PASSWORD = "0000"

# 관리자 로그인 계정 (사원 등록/관리 접근 제어). 실제 배포 시 반드시 변경.
ADMIN_ID = "0000"
ADMIN_PASSWORD = "0000"

# 관리자 회원가입 개방 여부. False면 로그인 화면의 회원가입이 막힌다(보안).
# 데모에서 새 계정을 만들어 보이려면 True로. (실운영 시 False 권장)
ADMIN_ALLOW_SIGNUP = False

# 청크 보관 기간(일). 이보다 오래된 녹화 청크는 자동 삭제. 0이면 자동삭제 안 함.
RETENTION_DAYS = 7

# 같은 비허가자 감지 이벤트를 이 간격(초) 내에는 중복 기록하지 않음(로그 폭주 방지).
DETECT_LOG_COOLDOWN = 10

# ArcFace 코사인 유사도 임계값
MATCH_THRESHOLD = 0.45
FACE_AUTH_RETAIN_THRESHOLD = 0.40
FACE_AUTH_TRACK_IOU = 0.35
# 기존 추적 안정화 기본값을 설정으로 노출한다. 값 자체는 변경하지 않는다.
FACE_AUTH_CONFIRM_MATCHES = 2
FACE_AUTH_CONFIRM_WINDOW = 3
FACE_AUTH_REVOKE_MISMATCHES = 5
FACE_AUTH_HOLD_SECONDS = 0.8

# Hailo-8L 가속: SCRFD 탐지는 Hailo에서, ArcFace 인식은 CPU에서 실행.
# True여도 hailo_platform이 없으면 자동으로 insightface(CPU)로 폴백.
# ⚠️ Hailo ArcFace는 임베딩 공간이 달라 전환 시 재등록 필요.
USE_HAILO = True
SCRFD_HEF_PATH = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
ARCFACE_HEF_PATH = "/usr/share/hailo-models/arcface_mobilefacenet.hef"
HAILO_DET_THRESH = 0.5
# Hailo 사용 불가 시 CPU SCRFD 폴백의 기존 임계값. 실장비 검증 없이
# 낮추지 않고 탐지 신뢰도와 ArcFace 유사도를 별도로 진단한다.
CPU_SCRFD_DET_THRESHOLD = 0.6

# (참고) zxc.txt 환경에서의 Hailo 모델 경로 변수명
HAILO_SCRFD_HEF   = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
HAILO_ARCFACE_HEF = "/usr/share/hailo-models/arcface_mobilefacenet.hef"

# True면 내부인(등록 사원)도 익명화 (전원 보호, 권한자만 복원).
# False면 외부인만 익명화하고 내부인은 신원 표시.
ANONYMIZE_ALL = True

# 실시간 화면 익명화 방식: 가벼운 모자이크.
REALTIME_ANON = "mosaic"

# 실시간 화면(모자이크 표시) 갱신 fps 상한.
# 높을수록 화면이 부드러움.
PROCESS_MAX_FPS = 20

# 이 시간 동안 새 보호 프레임이 없으면 LIVE/REC 상태를 중지 상태로 표시한다.
CAMERA_STALE_SECONDS = 3.0

# 레거시 PSF Recorder 호환 설정(현재 AES RawRecorder 경로에서 미사용).
SAVE_FPS = 15

# 레거시 PSF Recorder 호환 설정(현재 미사용).

# 레거시 PSF Recorder 호환 설정(현재 미사용).
# 0이면 자동(max(1, 코어수 ÷ 워커수)).

# (구) 프레임 스킵. 시간 기반 PROCESS_MAX_FPS를 쓰므로 1로 둠.
PROCESS_EVERY_N = 1

# recorder가 뒤처질 때 원본 프레임을 쌓아두는 큐 최대 크기(장).
# 클수록 프레임을 덜 버리지만 메모리↑ (JPEG 압축 저장, 장당 ~50KB).
FRAME_QUEUE_MAX = 10

# 처리 전 프레임 가로 해상도 축소(px).
# 0이면 원본. 탐지 속도↑.
PROCESS_WIDTH = 640

# 모든 RGB 카메라는 공통 V4L2 관리자로 자동 선택합니다.
# RealSense D455도 UVC 장치라 OpenCV(webcam)로 컬러 스트림을 받을 수 있음.
CAMERA_TYPE = "webcam"

# D457은 Depth/IR/RGB가 별도 V4L2 노드로 노출된다. 운영 경로에서는
# USB 인터페이스 설명이 RGB인 노드와 YUYV 포맷만 허용한다.
REQUIRE_REALSENSE_RGB = True
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 장치 번호 기본값 없음. CAMERA_DEVICE 환경변수 또는 V4L2 자동 탐색.
# 선택된 경로는 실행 중 공통 관리자가 저장합니다.
CAMERA_SELECTED_DEVICE = None

# 카메라 실패 시 폴백 동영상 (mp4 등).
# None이면 "Reconnecting" 재시도.
# 카메라 인식에만 집중 → 폴백 영상 사용 안 함.
VIDEO_FALLBACK = None
VIDEO_FALLBACK_FPS = 25   # 재생 속도 (원본 영상 fps에 맞춤)

# True이면 카메라 탐색을 건너뛰고 무조건 VIDEO_FALLBACK 영상을 사용.
# (이 PC 카메라가 검은 프레임만 줄 때 demo.mp4로 강제하는 용도)
FORCE_VIDEO = False

# 레거시 PSF Recorder 호환 설정(현재 미사용).
RECORD_INTERVAL = 0

# 청크 길이(초). monotonic 촬영 경과 시간을 기준으로 분할한다.
# 저장 계층: raw_data/YYYY-MM-DD/HH/HH-MM-SS/
CHUNK_SECONDS = 60  # 1분

# 파일·합성 테스트 경로용 프레임 청크 상한. 운영 카메라는 CHUNK_SECONDS만 사용.
# 운영 시간 청크에는 적용하지 않는다.
# 0이면 CHUNK_SECONDS * SAVE_FPS 로 자동 계산한다.
FRAMES_PER_CHUNK = 0

# 복원 영상 출력 fps (부드러움용). 실제 영상 길이는 프레임 타임스탬프로 맞춰짐.
RESTORE_VIDEO_FPS = 15

# 원격 복원 기능은 사용하지 않는다.
REMOTE_RESTORE_URL = None

MODAL_RESTORE_URL = None

# 원격 보호 기능은 사용하지 않는다.
REMOTE_PROTECT_URL = None

# [실험] 저장 시 여러 프레임을 한 번에 묶어 전송(배치). 1이면 기존 단일 전송.
# 왕복/포장 오버헤드를 줄여 저장 fps를 높일 수 있음. 4~8 정도로 실험.
# (병목이 GPU 연산이면 효과가 제한적일 수 있음)

# ffmpeg 실행 파일 경로 (None이면 PATH에서 탐색).
# 브라우저 호환 H.264 변환에 사용.
# winget 설치 시 PATH에 없을 수 있어 직접 지정.
#FFMPEG_PATH = r"C:\Users\HOSEO\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
FFMPEG_PATH = "ffmpeg"  # PATH에 있으면 그냥 ffmpeg로 됨


# AES 암호화 원본 저장 위치
RAW_ENCRYPT_DIR = str(RAW_DATA_DIR)  # legacy name; all active chunk storage is unified here
# ─── 디스크 I/O 최적화 (Ramdisk & SD 하이브리드 설정) ────────────────────────
if platform.system() == "Linux":
    # 라즈베리파이: 실시간 저장은 RAM, 영구 보관은 SD카드
    RECORD_RAM_DIR = "/dev/shm/SecureFace_recordings"
    RECORD_SD_DIR = "recordings"
else:
    # 윈도우/맥 (로컬 테스트용): 구별할 필요 없이 기존 폴더 사용
    RECORD_RAM_DIR = "recordings"
    RECORD_SD_DIR = "recordings"

# 폴더가 없으면 미리 생성
# Directory creation belongs to the active recorder, not configuration import.

# main.py selective recording (same server and monitor)
SELECTIVE_FPS = 10
SELECTIVE_CHUNK_FRAMES = 8
SELECTIVE_EXTERNAL_PASSWORD = "0001"
SELECTIVE_INTERNAL_PASSWORD = "0002"
FACE_AUTH_CONFIRM_MATCHES = 2
FACE_AUTH_CONFIRM_WINDOW = 3
FACE_AUTH_REVOKE_MISMATCHES = 5
FACE_AUTH_HOLD_SECONDS = 0.8
SELECTIVE_SOURCE = os.environ.get("SRX_VIDEO_SOURCE", "")  # Empty: CAMERA_DEVICE; otherwise a local video file
SELECTIVE_MODEL_DIR = os.path.expanduser("~/.insightface/models/buffalo_s")
SELECTIVE_DETECTOR_PATH = ""  # Optional explicit SCRFD ONNX file
SELECTIVE_RECOGNIZER_PATH = ""  # Optional explicit ArcFace ONNX file

# 선택적 보호 영상: 얼굴 크기에 비례하는 강한 블러 (축소 ROI에서 계산)
SELECTIVE_BLUR_MIN_KERNEL = 61
SELECTIVE_BLUR_FACE_RATIO = 0.9
SELECTIVE_BLUR_WORK_SIZE = 32
SELECTIVE_BLUR_EXPANSION_RATIO = 0.0
