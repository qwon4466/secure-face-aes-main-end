"""공통 카메라 관리자 진단. CAMERA_DEVICE 환경변수로 장치를 지정합니다."""
from core.camera_manager import CameraManager
import cv2



def main():
    cap = CameraManager().open()
    try:
        if not cap.isOpened():
            print(f"카메라 {cap.selected}를 열 수 없습니다.")
            return

        print(f"카메라 {cap.selected} 열림. q 키로 종료.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
