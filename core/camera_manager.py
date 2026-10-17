"""Single-owner Linux V4L2 discovery and validated color capture."""
import fcntl
import glob
import os
from pathlib import Path
import re
import threading
import cv2
import numpy as np

_OWNER = threading.Lock()
SELECTED = {}


def product(path):
    node = Path(path).resolve().name
    try:
        return Path('/sys/class/video4linux', node, 'name').read_text().strip()
    except OSError:
        return 'Unknown'


def interface_name(path):
    """Return the USB interface role (for example RealSense ``RGB``/``Depth``)."""
    node = Path(path).resolve().name
    try:
        return Path('/sys/class/video4linux', node, 'device', 'interface').read_text().strip()
    except OSError:
        return ''


def stream_index(path):
    """UVC video index; RealSense index 0 is image data and index 1 metadata."""
    node = Path(path).resolve().name
    try:
        return int(Path('/sys/class/video4linux', node, 'index').read_text().strip())
    except (OSError, ValueError):
        return None


def camera_role(path):
    """Classify sensor role without relying on the unstable /dev/video number."""
    name = product(path)
    interface = interface_name(path)
    combined = (name + ' ' + interface).lower()
    if 'realsense' in combined:
        if 'rgb' in interface.lower() or 'color' in interface.lower():
            return 'realsense-rgb' if stream_index(path) == 0 else 'realsense-metadata'
        if interface:
            return 'realsense-non-rgb'
        # Fail closed when a RealSense interface cannot be identified.  A
        # Depth/IR node converted by OpenCV can still look like a BGR frame.
        return 'realsense-unknown'
    return 'color'


def _fourcc(value):
    value = int(value)
    return ''.join(chr((value >> (8*i)) & 255) for i in range(4)).rstrip('\x00')


def candidates():
    def priority(path):
        role = camera_role(path)
        stable = 0 if '/by-path/' in path else (1 if '/by-id/' in path else 2)
        return (0 if role == 'realsense-rgb' else (1 if role == 'color' else 2), stable,
                int(re.search(r'video(\d+)$', str(Path(path).resolve())).group(1))
                if re.search(r'video(\d+)$', str(Path(path).resolve())) else 9999)
    links = sorted(glob.glob('/dev/v4l/by-path/*') + glob.glob('/dev/v4l/by-id/*'), key=priority)
    nodes = sorted(glob.glob('/dev/video*'), key=priority)
    # Stable aliases are retained even when prioritizing a RealSense color node.
    paths = []; seen = set()
    for path in links + nodes:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real); paths.append(path)
    return sorted(paths, key=lambda p: priority(p)[0])


class CameraManager:
    def __init__(self, device=None, config=None, stop_event=None):
        self.stop_event = stop_event
        self.lease = None
        self.requested = os.environ.get('CAMERA_DEVICE') or device
        self.config = config
        self.cap = None
        self.selected = None
        self.failures = []
        self.owned = False
        self.generation = 0
        self.required_role = ('realsense-rgb'
                              if bool(getattr(config, 'REQUIRE_REALSENSE_RGB', False))
                              else None)

    def open(self):
        if self.cap is not None:
            return self
        if not self.owned:
            if not _OWNER.acquire(blocking=False):
                raise RuntimeError('카메라가 이미 공통 관리자에서 사용 중입니다')
            self.owned = True
        if self.lease is None:
            try:
                self.lease = os.open(f'/tmp/secureface-camera-{os.getuid()}.lock',
                                     os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.release()
                raise RuntimeError('다른 프로세스가 공통 카메라 관리자를 사용 중입니다') from exc
        paths = ([self.requested] if self.requested else []) + candidates()
        seen = set(); self.failures = []
        for path in paths:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            if self._probe(path):
                return self
            if path == self.requested:
                print('[Camera] 지정 장치 실패: 자동 탐색으로 전환합니다', flush=True)
        self.release()
        raise RuntimeError('컬러 카메라 검색 실패: ' + ('; '.join(self.failures) or '검색된 장치 없음'))

    def _probe(self, path, native=False):
        cap = None
        requested = False
        try:
            if not Path(path).exists():
                raise ValueError('장치가 존재하지 않습니다')
            name = product(path)
            interface = interface_name(path)
            role = camera_role(path)
            if role in ('realsense-non-rgb', 'realsense-unknown', 'realsense-metadata'):
                raise ValueError(f'RealSense RGB 노드가 아닙니다 (interface={interface or "unknown"})')
            if self.required_role is not None and role != self.required_role:
                raise ValueError(f'필수 카메라 역할 불일치: {role} != {self.required_role}')
            cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
            if not cap.isOpened():
                raise ValueError('V4L2 열기 실패 (권한/점유/영상 기능 확인)')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            profile = 'device-native'
            if not native and role == 'realsense-rgb':
                requested = True
                profile = 'realsense-rgb-yuyv'
                width = int(getattr(self.config, 'CAMERA_WIDTH', 640))
                height = int(getattr(self.config, 'CAMERA_HEIGHT', 480))
                fps = float(getattr(self.config, 'CAMERA_FPS', 30))
                settings = ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')),
                            (cv2.CAP_PROP_FRAME_WIDTH, width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_FPS, fps))
                for prop, value in settings:
                    if not cap.set(prop, value):
                        raise ValueError('RealSense RGB YUYV 프로파일 설정 실패')
                print(f'[Camera] RealSense RGB 요청: {path} YUYV {width}x{height} @{fps:g}fps',flush=True)
            # Logitech development profile only; other devices keep their native mode.
            elif not native and ('046d' in (name + path).lower() or 'logitech' in name.lower()):
                requested = True
                profile = 'logitech-request'
                settings = ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                            (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                            (cv2.CAP_PROP_FPS, 30))
                for prop, value in settings:
                    if not cap.set(prop, value):
                        raise ValueError('요청 설정 MJPG 640x480 30fps 지원 확인 실패')
                print(f'[Camera] {path}: MJPG 640x480 30fps 요청; 실제 수신값으로 검증', flush=True)
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH) <= 0 or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) <= 0:
                raise ValueError('너비 또는 높이가 0')
            color = False
            for _ in range(8):
                if self.stop_event is not None and self.stop_event.is_set():
                    raise ValueError("서버 종료 요청")
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    raise ValueError('연속 8프레임 읽기 실패')
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError('8비트 BGR 컬러 영상이 아닙니다')
                sample = frame[::8, ::8].astype(np.int16)
                color |= bool(np.mean(np.max(sample, axis=2)-np.min(sample, axis=2)) > 2 and np.mean(sample.std(axis=(0, 1))) > 2)
            if not color:
                raise ValueError('단색/깊이 영상 또는 컬러 확인 불가 (조명/장면 확인)')
            pixel_format = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            if role == 'realsense-rgb' and pixel_format != 'YUYV':
                raise ValueError(f'RealSense RGB 포맷 불일치: {pixel_format or "unknown"} != YUYV')
            self.cap = cap
            self.selected = {'device':str(path), 'product':name, 'width':int(frame.shape[1]),
                             'height':int(frame.shape[0]), 'fps':float(cap.get(cv2.CAP_PROP_FPS)),
                             'fourcc':pixel_format, 'pixel_format':pixel_format,
                             'profile':(profile if not native else
                                        ('realsense-rgb-native-verified'
                                         if role == 'realsense-rgb' else 'device-native')),
                             'role':role, 'interface':interface,
                             'node':str(Path(path).resolve()), 'generation':self.generation+1}
            if role == 'realsense-rgb':
                self.required_role = role
            SELECTED.clear(); SELECTED.update(self.selected)
            if self.config is not None:
                self.config.CAMERA_SELECTED_DEVICE = str(path)
            self.generation += 1
            print('[Camera] 선택: ' + str(self.selected), flush=True)
            return True
        except Exception as exc:
            if cap is not None:
                cap.release()
            if requested:
                retry=('RealSense 기본 모드를 같은 RGB 규칙으로 재검증'
                       if role == 'realsense-rgb' else '장치 기본 형식/해상도로 재시도')
                print(f'[Camera] {path}: 요청 모드 실패 ({exc}) → {retry}', flush=True)
                return self._probe(path, native=True)
            message = f'{path} ({product(path)}): {exc}' 
            self.failures.append(message)
            print('[Camera] 제외: ' + message, flush=True)
            return False

    def reconnect(self):
        previous = self.selected['device'] if self.selected else None
        if self.cap is not None:
            self.cap.release(); self.cap = None
        if not self.owned:
            return self.open()
        if previous and self._probe(previous):
            return self
        print('[Camera] 동일 장치 재연결 실패 → 전체 재검색', flush=True)
        return self.open()

    def read(self):
        try:
            return self.cap.read() if self.cap is not None else (False, None)
        except cv2.error as exc:
            print('[Camera] 읽기 오류:', exc, flush=True)
            return False, None

    def get(self, prop):
        return self.cap.get(prop) if self.cap is not None else 0

    def set(self, prop, value):
        return self.cap.set(prop, value) if self.cap is not None else False

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release(); self.cap = None
        if self.lease is not None:
            os.close(self.lease); self.lease = None
        if self.owned:
            self.owned = False; _OWNER.release()
