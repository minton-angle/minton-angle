import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_0to1_scaled_points(landmarks, canvas_w, canvas_h):
    """
    모든 키포인트를 [0, 1] 범위로 스케일링하고 캔버스 픽셀 좌표로 반환
    """
    # 1. 원점 설정: 엉덩이 중심
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    
    # 2. 기준 거리: 키 (코 ~ 발목 중심)
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    person_height = abs(landmarks[0].y - ankle_y)
    if person_height == 0: person_height = 0.1

    points_px = []
    for lm in landmarks:
        # 3. 엉덩이 기준 상대 좌표 계산 후 키 비율로 변환
        rel_x = (lm.x - hip_x) / person_height
        rel_y = (lm.y - hip_y) / person_height
        
        # 4. [0, 1] 범위로 매핑 (0.5가 엉덩이 위치)
        # 0.35는 스켈레톤이 캔버스에 적당한 크기(약 70%)로 채워지게 하는 배율입니다.
        norm_x = 0.5 + (rel_x * 0.35)
        norm_y = 0.5 + (rel_y * 0.35)
        
        # 5. 경계값 제한 (0.0 ~ 1.0)
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)
        
        # 6. 시각화를 위한 픽셀 좌표 변환
        px_x = int(norm_x * canvas_w)
        px_y = int(norm_y * canvas_h)
        points_px.append((px_x, px_y))
        
    return points_px

# ---------------------------------------------------------
# 경로 설정 (파일 이름에 맞게 수정하세요)
video_paths = ['/Users/minji/Documents/pro_swing_1_GT.mp4', '/Users/minji/Documents/pro_swing_2_GT.mp4', '/Users/minji/Documents/Baekcoach.mp4'] 
output_path = '/Users/minji/Documents/minton-angle/backend/data/standard/combined_comparison_fixed_height_2.mp4'

caps = [cv2.VideoCapture(p) for p in video_paths]
fps = int(caps[0].get(cv2.CAP_PROP_FPS))
c_w, c_h = 400, 400  # 각 스켈레톤이 그려질 개별 정사각형 캔버스 크기

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (c_w * 3, c_h))

print("영상을 [0, 1] 범위로 정규화하여 통합 중입니다...")

while True:
    canvases = []
    all_success = True
    
    for cap in caps:
        success, frame = cap.read()
        if not success:
            all_success = False
            break
            
        # 개별 검은색 캔버스 생성
        canvas = np.zeros((c_h, c_w, 3), dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 0~1 스케일링된 픽셀 좌표 가져오기
            px_points = get_0to1_scaled_points(results.pose_landmarks.landmark, c_w, c_h)
            
            # 스켈레톤 선 그리기
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(canvas, px_points[start_idx], px_points[end_idx], (0, 255, 0), 2)
            
            # 관절 점 그리기
            for pt in px_points:
                cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
        
        canvases.append(canvas)
        
    if not all_success: break

    # 3개 캔버스를 가로로 결합
    combined_frame = np.hstack(canvases)
    
    out.write(combined_frame)
    cv2.imshow('0-1 Normalized Comparison', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == 27: break

for cap in caps: cap.release()
out.release()
cv2.destroyAllWindows()
print(f"저장 완료: {output_path}")