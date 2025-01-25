import time
import serial
import threading
import os
import cv2
import numpy as np
from datetime import datetime

# --------------------- 사용자 환경에 맞게 수정 --------------------- #
SERIAL_PORT = "/dev/ttyACM0"    # 아두이노(또는 다른 MC)와 연결된 시리얼 포트
BAUD_RATE = 9600                # 통신 속도
IMAGE_SAVE_DIR = "captured_images"
# ------------------------------------------------------------------ #

# 글로벌 이벤트: IR 센서 감지 시 'True'로 세팅됨
ir_triggered_event = threading.Event()

# 종료 플래그
terminate_event = threading.Event()

# 시리얼 포트 오픈
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial port {SERIAL_PORT} opened successfully.")
except serial.SerialException as e:
    print(f"Error opening serial port {SERIAL_PORT}: {e}")
    exit(-1)

# 카메라 초기화
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Cannot open camera.")
    exit(-1)

# 이미지 저장 폴더 생성
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------------
# PCB 영역을 HSV 색공간에서 찾아 크롭하는 함수 (예: 녹색 PCB)
# ------------------------------------------------------------------------------------
def detect_and_crop_pcb(frame):
    """
    입력 프레임에서 'PCB 기판 전체'가 보이도록 크롭하는 함수.
    1) HSV 변환 및 녹색(PCB) 범위 마스킹
    2) morphology 연산으로 노이즈 제거
    3) 가장 큰 컨투어 찾기
    4) boundingRect로 크롭 (PCB 전체)
    실제 녹색 PCB 색상에 맞춰 HSV 범위를 적절히 조정해야 함.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 환경에 맞게 녹색 범위를 조절
    lower_green = np.array([35, 40, 40])  
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_morphed = cv2.morphologyEx(mask_morphed, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for PCB. Check color range or lighting.")
        return None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    if w < 20 or h < 20:
        print("Detected PCB contour too small. Possibly noise.")
        return None

    cropped_pcb = frame[y:y+h, x:x+w]
    return cropped_pcb

# ------------------------------------------------------------------------------------
# 밝기(조도) 낮추기: beta에 음수값을 줘서 전체적으로 어둡게
# ------------------------------------------------------------------------------------
def lower_brightness(image, delta=30):
    """
    이미지 전체의 밝기를 낮추는 함수.
    alpha=1.0, beta=-delta → beta가 음수이면 어둡게 처리.
    """
    # convertScaleAbs: alpha * pixel + beta 후, 0~255로 클램핑
    lowered = cv2.convertScaleAbs(image, alpha=1.0, beta=-delta)
    return lowered

# ------------------------------------------------------------------------------------
# 시리얼 스레드
# ------------------------------------------------------------------------------------
def serial_read_thread():
    """
    IR 센서 감지 신호를 기다리는 스레드.
    시리얼에서 '0' 바이트를 수신하면 IR 이벤트를 트리거한다.
    """
    while not terminate_event.is_set():
        try:
            data = ser.read()  # timeout=1 로 설정되어 있으므로 블로킹되지 않음
            if data:
                # IR 센서로부터 '0' 이 들어온다고 가정
                if data == b"0":
                    print("IR Sensor Triggered!")
                    ir_triggered_event.set()
        except serial.SerialException as e:
            print(f"Serial read error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error in serial thread: {e}")
            break
        
        time.sleep(0.01)

# ------------------------------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------------------------------
def main():
    """
    1) 카메라 영상 표시 
    2) IR 감지 시나리오(컨베이어 0.5초 후 정지 -> 캡처 -> PCB 크롭 -> 밝기 낮춰서 저장 -> 재개).
    """
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Warning: Failed to read frame from camera.")
                continue

            # 현재 프레임을 윈도우에 띄움
            cv2.imshow("Live Feed", frame)

            # 'q' 키를 누르면 프로그램 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit signal received (q key).")
                break

            # IR 센서 감지 이벤트 발생 시나리오
            if ir_triggered_event.is_set():
                print("IR 이벤트 처리 중...")

                # 1) 0.5초 대기
                time.sleep(0.5)

                # 2) 컨베이어 정지 명령 ('0')
                print("Stop conveyor.")
                ser.write(b"0")

                # 3) 컨베이어 정지 후 이미지 캡처
                ret_capture, frame_capture = cam.read()
                if ret_capture:
                    # PCB 크롭
                    cropped_pcb = detect_and_crop_pcb(frame_capture)
                    if cropped_pcb is not None:
                        # 조도(밝기) 낮추기
                        dark_pcb = lower_brightness(cropped_pcb, delta=40)  # delta 값은 테스트 통해 조정

                        # 저장
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename_dark = os.path.join(IMAGE_SAVE_DIR, f"pcb_dark_{timestamp}.jpg")
                        cv2.imwrite(filename_dark, dark_pcb)
                        print(f"Cropped (Darkened) PCB Image saved: {filename_dark}")
                    else:
                        print("No valid PCB detected for cropping.")
                else:
                    print("Failed to capture image after stopping conveyor.")

                # 추가 대기 (예: 1.2초)
                time.sleep(1.2)

                # 4) 컨베이어 동작 명령 ('1')
                print("Resume conveyor.")
                ser.write(b"1")

                # 이벤트 플래그 해제(초기화)
                ir_triggered_event.clear()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting.")
    finally:
        # 종료 신호
        terminate_event.set()
        ser.close()
        cam.release()
        cv2.destroyAllWindows()
        print("Resources released. Program terminated.")

# ------------------------------------------------------------------------------------
# 실행 구문
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 시리얼 수신용 스레드 시작 (IR 센서 감지 대기)
    serial_thread = threading.Thread(target=serial_read_thread, daemon=True)
    serial_thread.start()
    
    print("System is running. Press 'q' in the video window to exit.")
    
    # 메인 함수 실행
    main()