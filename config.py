"""
SecureFace-RX 전역 설정
"""

import os
import platform
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
# ─── 통합 시스템 설정 (123.txt 우선) ───────────────────────────────
# 관리자 로그인 계정 (사원 등록/관리 접근 제어). 실제 배포 시 반드시 변경.
ADMIN_ID = "0000"
ADMIN_PASSWORD = "0000"

# 관리자 회원가입 개방 여부. False면 로그인 화면의 회원가입이 막힌다(보안).
# 데모에서 새 계정을 만들어 보이려면 True로. (실운영 시 False 권장)
ADMIN_ALLOW_SIGNUP = False

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

# 이 시간 동안 새 보호 프레임이 없으면 LIVE/REC 상태를 중지 상태로 표시한다.
CAMERA_STALE_SECONDS = 3.0

# 처리 전 프레임 가로 해상도 축소(px).
# 0이면 원본. 탐지 속도↑.
PROCESS_WIDTH = 640

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

# True이면 카메라 탐색을 건너뛰고 무조건 VIDEO_FALLBACK 영상을 사용.
# (이 PC 카메라가 검은 프레임만 줄 때 demo.mp4로 강제하는 용도)
FORCE_VIDEO = False

# 청크 길이(초). monotonic 촬영 경과 시간을 기준으로 분할한다.
# 저장 계층: raw_data/YYYY-MM-DD/HH/HH-MM-SS/
CHUNK_SECONDS = 60  # 1분


# AES 암호화 원본 저장 위치
RAW_ENCRYPT_DIR = str(RAW_DATA_DIR)  # legacy name; all active chunk storage is unified here
# main.py selective recording (same server and monitor)
SELECTIVE_FPS = 10
SELECTIVE_CHUNK_FRAMES = 8
SELECTIVE_EXTERNAL_PASSWORD = "0001"
SELECTIVE_INTERNAL_PASSWORD = "0002"
FACE_AUTH_CONFIRM_MATCHES = 2
FACE_AUTH_CONFIRM_WINDOW = 3
FACE_AUTH_REVOKE_MISMATCHES = 5
FACE_AUTH_HOLD_SECONDS = 0.8
# 비허가자 감지 알림: 이 시간 동안 계속 비허가일 때만 알림(등록자 인식 확정 전 오탐 방지),
# 같은 자리에서 이 시간 안에 다시 잡힌 얼굴은 같은 사람으로 보고 다시 알리지 않음
DETECT_ALERT_CONFIRM_SECONDS = 1.0
DETECT_ALERT_REARM_SECONDS = 5.0
SELECTIVE_SOURCE = os.environ.get("SRX_VIDEO_SOURCE", "")  # Empty: CAMERA_DEVICE; otherwise a local video file
SELECTIVE_MODEL_DIR = os.path.expanduser("~/.insightface/models/buffalo_s")
SELECTIVE_DETECTOR_PATH = ""  # Optional explicit SCRFD ONNX file
SELECTIVE_RECOGNIZER_PATH = ""  # Optional explicit ArcFace ONNX file

# 선택적 보호 영상: 얼굴 크기에 비례하는 강한 블러 (축소 ROI에서 계산)
SELECTIVE_BLUR_MIN_KERNEL = 61
SELECTIVE_BLUR_FACE_RATIO = 0.9
SELECTIVE_BLUR_WORK_SIZE = 32
SELECTIVE_BLUR_EXPANSION_RATIO = 0.0
