import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame, reference_height=500, canvas_size=(800, 600)):
    """
    프레임에서 포즈를 추출하고 고정된 키(reference_height)에 맞춰 정규화된 스켈레톤 반환
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 1. 기준점 및 현재 키 계산
        hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_center_y = (landmarks[23].y + landmarks[24].y) / 2
        ankle_center_y = (landmarks[27].y + landmarks[28].y) / 2
        
        # 현재 영상 속 사람의 픽셀 높이 비율
        current_height_ratio = abs(landmarks[0].y - ankle_center_y)
        if current_height_ratio == 0: current_height_ratio = 0.1

        points = []
        for lm in landmarks:
            # 2. 정규화: (좌표 - 원점) / 현재 키 * 고정할 키
            # 이렇게 하면 모든 영상의 키가 reference_height 픽셀로 통일됨
            norm_x = (lm.x - hip_center_x) / current_height_ratio * reference_height
            norm_y = (lm.y - hip_center_y) / current_height_ratio * reference_height
            
            # 캔버스 중앙 하단에 배치
            display_x = int(norm_x + canvas_size[0] // 2)
            display_y = int(norm_y + canvas_size[1] // 2 + 100)
            points.append((display_x, display_y))

        # 3. 스켈레톤 그리기
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if 0 <= points[start_idx][0] < canvas_size[0] and 0 <= points[end_idx][0] < canvas_size[0]:
                cv2.line(canvas, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
            
    return canvas

# ---------------------------------------------------------
# 설정 (경로를 본인 환경에 맞게 수정하세요)
# ---------------------------------------------------------
video_paths = [
    '/Users/minji/Documents/pro_swing_1_GT.mp4', 
    '/Users/minji/Documents/pro_swing_2_GT.mp4', 
    '/Users/minji/Documents/Baekcoach.mp4'
]
output_path = '/Users/minji/Documents/minton-angle/backend/data/standard/combined_comparison_fixed_height.mp4'

caps = [cv2.VideoCapture(p) for p in video_paths]
fps = int(caps[0].get(cv2.CAP_PROP_FPS))
single_canvas_w, single_canvas_h = 400, 600 # 개별 영상 크기

# 결과 영상: 가로로 3개 배치 (400*3, 600)
combined_w = single_canvas_w * 3
combined_h = single_canvas_h

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

print("영상을 통합 처리 중입니다...")

while True:
    frames = []
    all_success = True
    
    for cap in caps:
        success, frame = cap.read()
        if not success:
            all_success = False
            break
        # 고정된 키 400px 기준으로 스켈레톤 추출
        processed = process_frame(frame, reference_height=350, canvas_size=(single_canvas_w, single_canvas_h))
        frames.append(processed)
        
    if not all_success: break

    # 3개 영상을 가로로 합치기
    combined_frame = np.hstack(frames)
    
    out.write(combined_frame)
    cv2.imshow('3-Way Comparison (Fixed Height)', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == 27: break

for cap in caps: cap.release()
out.release()
cv2.destroyAllWindows()
print(f"작업 완료! 저장 경로: {output_path}")