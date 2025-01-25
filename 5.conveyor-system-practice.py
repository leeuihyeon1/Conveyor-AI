import time
import serial
import requests
import numpy
import threading
import cv2
from datetime import datetime
import sqlite3
from flask import Flask, Response, render_template
import json
from requests.auth import HTTPBasicAuth

# 플라스크 서버 설정
app = Flask(__name__)

# 시리얼 통신 설정
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

# API endpoint 설정

# L6 모델
#api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/2ab8de9c-0725-4166-a540-b4d6557fdf02/inference"
# L 모델
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/f43c114c-c451-4571-aa18-0107428a16e4/inference"
# M 모델
#api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/b2a6115e-1487-4613-8419-96563878b36b/inference"

ACCESS_KEY = "9yIZaEry9A2bzfT3hpx1l3yZLllSESnH49kgUyei"

# 이벤트 설정
ir_triggered_event = threading.Event()
terminate_event = threading.Event()

# 정상 제품과 불량 제품의 기준 추가
target_counts_normal = {
    'RASBERRY PICO': 1,
    'HOLE': 4, 
    'CHIPSET': 1,
    'USB': 1,
    'OSCILLATOR': 1,
    'BOOTSEL': 1
}


# 공유 데이터를 위한 전역 딕셔너리
shared_data = {
    'current_frame': None,
    'detection_frame': None,
    'status': None
}

def crop_img(img, size_dict):    
    x = 98      # 왼쪽에서 150픽셀 지점부터 시작
    y = 40       # 위에서 20픽셀 지점부터 시작
    w = 450      # 너비 400픽셀로 감소
    h = 900     # 높이 유지
    img = img[y : y + h, x : x + w]
    return img


def process_frame(img, result):
    """프레임에 바운딩 박스와 정보를 그리는 함수"""
    # 객체 카운트 초기화
    object_counts = {}
    
    # 색상 매핑 정의
    colors = {
        'RASBERRY PICO': (0, 255, 0),    # 초록색
        'HOLE': (255, 0, 0),             # 파란색 
        'CHIPSET': (0, 0, 255),          # 빨간색
        'BOOTSEL': (255, 255, 0),        # 청록색
        'USB': (255, 0, 255),            # 분홍색
        'OSCILLATOR': (128, 0, 128),     # 보라색
        'BROKEN_USB': (0, 128, 128)      # 청록색
    }
    
    # 객체 카운트 계산
    for obj in result['objects']:
        if obj['score'] > 0.5:
            class_name = obj['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

    
    # 바운딩 박스와 레이블 그리기
    for obj in result['objects']:
        if obj['score'] > 0.5:
            start_point = (obj['box'][0], obj['box'][1])
            end_point = (obj['box'][2], obj['box'][3])
            class_name = obj['class']
            color = colors.get(class_name, (255,255,255))
            
            # 바운딩 박스 그리기
            cv2.rectangle(img, start_point, end_point, color, 1)
            
            # 클래스명과 신뢰도 표시
            text = f"{class_name} ({obj['score']:.2f})"
            position = (obj['box'][0], obj['box'][1] - 10)
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
    
    # 제품 상태 판정
    is_normal = True
    is_broken = False
    
    # 정상 제품 기준 확인
    for class_name, target_count in target_counts_normal.items():
        if object_counts.get(class_name, 0) != target_count:
            is_normal = False
            break
    
    # 불량 제품 확인 - BROKEN_USB 또는 SLICE_RASBERRY가 있으면 불량으로 처리
    if 'BROKEN_USB' in object_counts or 'SLICE_RASBERRY' in object_counts:
        is_broken = True
    
    # 판정 결과 표시
    if is_normal:
        status = "정상"
    elif is_broken:
        status = "불량"
    else:
        status = "재확인"
    
    return img, status


def inference_reqeust(img: numpy.array, api_url: str):
    """이미지를 API로 전송하여 추론을 요청합니다."""
    _, img_encoded = cv2.imencode(".jpg", img)
    
    response = requests.post(
        url=api_url,
        auth=HTTPBasicAuth("kdt2024_1-19", ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=img_encoded.tobytes(),
    )
    
    return response.json()


def serial_read_thread(ser):
    """IR 센서 감지 신호를 기다리는 스레드"""
    while not terminate_event.is_set():
        try:
            data = ser.read()
            if data:
                print(f"수신된 데이터: {data}")  # 실제로 어떤 데이터가 오는지 확인
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


def create_database():
    """데이터베이스 테이블 생성"""
    try:
        conn = sqlite3.connect('product_quality.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS 제품품질 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                상태 TEXT NOT NULL,
                불량유형 TEXT,
                객체수량 TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("데이터베이스 테이블 생성 완료")
    except sqlite3.Error as e:
        print(f"데이터베이스 생성 오류: {e}")

def save_quality_data(status, object_counts):
    """품질 데이터를 데이터베이스에 저장"""
    try:
        # 매번 새로운 연결 생성
        conn = sqlite3.connect('product_quality.db')
        cursor = conn.cursor()
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 불량 유형 확인
        defect_type = None
        if status == "불량" or status == "재확인":  # 재확인 상태도 포함
            if 'BROKEN_USB' in object_counts:
                defect_type = "USB 손상"
            elif 'SLICE_RASBERRY' in object_counts:  # SLICE_RASBERRY 케이스 추가
                defect_type = "라즈베리 손상"
        
        # 객체 수량을 문자열로 변환
        counts_str = ", ".join([f"{k}:{v}" for k, v in object_counts.items()])
        
        cursor.execute('''
            INSERT INTO 제품품질 (datetime, 상태, 불량유형, 객체수량)
            VALUES (?, ?, ?, ?)
        ''', (current_time, status, defect_type, counts_str))
        
        conn.commit()
        conn.close()
        print(f"품질 데이터 저장 완료: {status}")
        
    except sqlite3.Error as e:
        print(f"데이터 저장 오류: {e}")
        if 'conn' in locals():
            conn.close()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while not terminate_event.is_set():
        if shared_data['current_frame'] is not None:
            # JPEG 품질 설정 (0-100, 낮을수록 더 압축됨)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', shared_data['current_frame'], encode_param)
    
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def gen_detection():
    while not terminate_event.is_set():
        if shared_data['detection_frame'] is not None:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', shared_data['detection_frame'], encode_param)
            
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_feed')
def detection_feed():
    return Response(gen_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return json.dumps({'status': shared_data['status'] if shared_data['status'] else '대기 중...'})

def main():
    # 데이터베이스 테이블 생성
    create_database()
    
    # 시리얼 포트 초기화
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial port {SERIAL_PORT} opened successfully.")
        
        # 시리얼 통신 스레드 시작
        serial_thread = threading.Thread(target=serial_read_thread, args=(ser,), daemon=True)
        serial_thread.start()
        
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        return

    def process_video():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        try:
            while not terminate_event.is_set():
                ret, img = cap.read()
                if not ret:
                    continue

                shared_data['current_frame'] = img
                
                if ir_triggered_event.is_set():
                    print("IR 이벤트 처리 중...")
                    start_time = time.time()
                    
                    # 대기 시간 0.3초에서 0.15초로 감소
                    time.sleep(0.15)  # 컨베이어 정지 전 대기 시간 단축
                    
                    # 컨베이어 정지
                    ser.write(b"0")
                    
                    # 안정화 대기 시간 0.05초에서 0.02초로 감소
                    time.sleep(0.02)
                    
                    # 이미지 캡처 및 처리
                    ret, img = cap.read()
                    if ret:
                        # 이미지 크기를 더 작게 조정하여 처리 속도 향상
                        processed_img = cv2.resize(img, (480, 360))  # 더 작은 크기로 조정
                        processed_img = crop_img(processed_img, None)
                        
                        # API 요청 및 처리
                        result = inference_reqeust(processed_img, api_url)
                        
                        if result:
                            processed_img, status = process_frame(processed_img, result)
                            # 결과 이미지 크기 조정하여 네트워크 부하 감소
                            detection_frame = cv2.resize(processed_img, (640, 480))
                            shared_data['detection_frame'] = detection_frame
                            shared_data['status'] = status
                            
                            object_counts = {}
                            for obj in result['objects']:
                                if obj['score'] > 0.5:
                                    class_name = obj['class']
                                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                            save_quality_data(status, object_counts)
                    
                    # 안정화 대기 시간 0.3초에서 0.15초로 감소
                    time.sleep(0.15)
                    ser.write(b"1")
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"처리 시간: {total_time:.2f}초")
                    
                    ir_triggered_event.clear()
                
                # 메인 루프 대기 시간도 감소
                time.sleep(0.02)  # 0.03초에서 0.02초로 감소
                
        finally:
            cap.release()
    
    # 비디오 처리 스레드 시작
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    # Flask 서버 시작
    app.run(host='0.0.0.0', port=5000)
    
    # 종료 처리
    terminate_event.set()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("시스템 시작. 웹 브라우저에서 http://localhost:5000 에 접속하세요.")
    main()