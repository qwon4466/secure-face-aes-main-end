"""
옛 CameraProcessor(INN 익명화 시절)에서 코사인 유사도 함수만 남긴 파일.

core/selective_recognition.legacy_components()가 이 파일을 import하지 않고 AST로
읽어 _cosine_sim의 구조 해시를 검사한 뒤 사용한다. 함수 본문을 바꾸면 해시 검사에서
'cosine 함수 변경'으로 실행이 중단되므로 수정하지 말 것.
"""
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0
