"""Shared strong anonymization and Korean status overlays."""
from functools import lru_cache
from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

@lru_cache(maxsize=8)
def korean_font(size=18):
    paths=[os.environ.get('CAMERA_LABEL_FONT',''),
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
           '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf']
    for path in paths:
        if path and Path(path).is_file():
            print('[FaceLabel] 한글 글꼴:',path,flush=True)
            return ImageFont.truetype(path,size)
    message='한글 글꼴 없음: Noto Sans CJK/Nanum 설치 또는 CAMERA_LABEL_FONT 지정 필요'
    print('[FaceLabel] 오류:',message,flush=True)
    raise RuntimeError(message)

def annotate(frame,faces):
    font=korean_font(); image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(image);w,h=image.size
    for face in faces:
        # allowed: 시간·층 출입 판정. 판정이 없는 얼굴은 인식 결과로 표시한다.
        allowed=face.get('allowed',face.get('authorized',face.get('group')=='internal'))
        reason=face.get('access_reason','')
        text='허가자' if allowed else ('비허가자 · '+reason if reason and reason!='미등록' else '비허가자')
        color=(0,255,0) if allowed else (255,0,0)
        x1,y1,x2,y2=face['bbox']
        draw.rectangle((x1,y1,min(w-1,x2),min(h-1,y2)),outline=color,width=2)
        box=draw.textbbox((0,0),text,font=font);tw,th=box[2]-box[0]+8,box[3]-box[1]+8
        # Anchor the label immediately above the bbox's upper-left corner.
        x=max(0,min(x1,w-tw));y=max(0,min(y1-th,h-th))
        draw.rectangle((x,y,min(w-1,x+tw),min(h-1,y+th)),fill=color)
        draw.text((x+4-box[0],y+4-box[1]),text,font=font,fill=(0,0,0) if allowed else (255,255,255))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
