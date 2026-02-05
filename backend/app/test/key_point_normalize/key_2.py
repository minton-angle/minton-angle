import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_normalized_points_by_shoulder(landmarks, canvas_size=(800, 800)):
    # 1. 어깨 좌표 추출
    l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    
    # 2. 기준 거리 계산: 어깨너비 (Shoulder Width)
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
    
    # 예외 처리 (어깨가 검출되지 않을 경우 대비)
    if shoulder_width == 0: shoulder_width = 0.1 
    
    # 3. 원점 설정: 양 어깨의 중심 (Shoulder Center)
    shoulder_center = (l_shoulder + r_shoulder) / 2
    
    normalized_landmarks = []
    for lm in landmarks:
        # 4. 정규화: (현재좌표 - 어깨중심) / 어깨너비
        norm_x = (lm.x - shoulder_center[0]) / shoulder_width
        norm_y = (lm.y - shoulder_center[1]) / shoulder_width
        
        # 5. 시각화를 위해 캔버스 좌표로 변환
        # 어깨너비가 캔버스에서 약 200픽셀 정도로 보이게 설정 (배율 200)
        display_x = int(norm_x * 200 + canvas_size[0] // 2)
        display_y = int(norm_y * 200 + canvas_size[1] // 2)
        
        normalized_landmarks.append((display_x, display_y))
        
    return normalized_landmarks

# ---------------------------------------------------------
# 파일 경로 설정
input_path = '/Users/minji/Documents/Baekcoach.mp4'  # 입력 영상 파일명
output_path = '/Users/minji/Documents/minton-angle/backend/data/standard/Baek_skeleton_2.mp4'  # 저장할 파일명

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
        norm_points = get_normalized_points_by_shoulder(results.pose_landmarks.landmark, (width, height))

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