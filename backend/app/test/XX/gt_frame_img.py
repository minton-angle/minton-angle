import cv2
import mediapipe as mp
import os

# --- 1. 설정: 경로 및 포인트 ---
VIDEO_PATH = '/Users/minji/Documents/pro_swing_3_GT.mp4'  # 원본 영상 경로
FRAME_OUTPUT_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_3_extracted_frames' # 프레임 이미지가 저장될 폴더
if not os.path.exists(FRAME_OUTPUT_DIR): os.makedirs(FRAME_OUTPUT_DIR)

# 19개 핵심 포인트
SELECTED_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. 영상 로드 ---
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"이미지 추출 시작: {FRAME_OUTPUT_DIR} 폴더를 확인하세요.")

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # 1. 스켈레톤 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 2. 선택한 19개 포인트 강조 (노란색 점)
        landmarks = results.pose_landmarks.landmark
        for idx in SELECTED_INDICES:
            lm = landmarks[idx]
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # 3. 프레임 번호를 파일명으로 하여 이미지 저장 (04d는 0001, 0002 처럼 4자리 숫자로 저장함)
    file_name = f"frame_{frame_idx:04d}.jpg"
    file_path = os.path.join(FRAME_OUTPUT_DIR, file_name)
    cv2.imwrite(file_path, frame)

    if frame_idx % 30 == 0:
        print(f"{frame_idx}번째 프레임 저장 중...")
    
    frame_idx += 1

cap.release()
pose.close()
print(f"🎉 총 {frame_idx}개의 이미지가 '{FRAME_OUTPUT_DIR}' 폴더에 저장되었습니다.")