import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_normalized_points(landmarks, canvas_size=(800, 800)):
    # 1. 기준점 설정: 엉덩이 중심 (Mid-hip)
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_center_y = (landmarks[23].y + landmarks[24].y) / 2
    
    # 2. 기준 거리(키) 계산: 코(0)와 양쪽 발목(27, 28)의 평균 지점
    ankle_center_x = (landmarks[27].x + landmarks[28].x) / 2
    ankle_center_y = (landmarks[27].y + landmarks[28].y) / 2
    
    person_height = np.sqrt((landmarks[0].x - ankle_center_x)**2 + (landmarks[0].y - ankle_center_y)**2)
    
    # 키 값이 0일 경우(검출 오류)를 대비한 예외 처리
    if person_height == 0: person_height = 1
    
    normalized_landmarks = []
    for lm in landmarks:
        norm_x = (lm.x - hip_center_x) / person_height
        norm_y = (lm.y - hip_center_y) / person_height
        
        # 캔버스 중앙을 기준으로 스케일업 (배율 400)
        display_x = int(norm_x * 400 + canvas_size[0] // 2)
        display_y = int(norm_y * 400 + canvas_size[1] // 2)
        normalized_landmarks.append((display_x, display_y))
        
    return normalized_landmarks

# ---------------------------------------------------------
# 파일 경로 설정
input_path = '/Users/minji/Documents/Baekcoach.mp4'  # 입력 영상 파일명
output_path = '/Users/minji/Documents/minton-angle/backend/data/standard/Baek_skeleton.mp4'  # 저장할 파일명

cap = cv2.VideoCapture(input_path)

# 영상 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = 800  # 저장할 캔버스 가로 크기
height = 800 # 저장할 캔버스 세로 크기

# 비디오 저장 설정 (Codec: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"영상 처리를 시작합니다: {input_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # RGB 변환 및 MediaPipe 처리
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 검은색 배경 캔버스 생성
    black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if results.pose_landmarks:
        # 정규화 좌표 계산
        norm_points = get_normalized_points(results.pose_landmarks.landmark, (width, height))

        # 1. 연결선 그리기 (초록색)
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            # 좌표가 캔버스 범위 내에 있는지 확인 후 그리기
            cv2.line(black_canvas, norm_points[start_idx], norm_points[end_idx], (0, 255, 0), 2)

        # 2. 키포인트 점 그리기 (빨간색)
        for pt in norm_points:
            cv2.circle(black_canvas, pt, 4, (0, 0, 255), -1)

    # 결과 프레임 파일에 저장
    out.write(black_canvas)

    # 실시간 모니터링 (선택 사항)
    cv2.imshow('Processing...', black_canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"저장이 완료되었습니다: {output_path}")