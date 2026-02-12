import cv2
import mediapipe as mp
import csv
import os

# --- 1. 설정: 경로 및 포인트 ---
VIDEO_PATH = '/Users/minji/Documents/pro_swing_3_GT.mp4'  # 원본 영상 경로
OUTPUT_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard'        # 결과물을 모을 폴더 이름
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 파일명 설정 (폴더 안에 저장되도록)
csv_save_path = os.path.join(OUTPUT_DIR, 'pro_swing_3_data.csv')
video_save_path = os.path.join(OUTPUT_DIR, 'pro_swing_3_analyzed_video.mp4')

# 19개 핵심 포인트
SELECTED_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. 영상 로드 및 저장 설정 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 영상 저장을 위한 설정 (코덱: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

# --- 3. CSV 준비 ---
header = ['frame']
for idx in SELECTED_INDICES:
    header.extend([f'{idx}_x', f'{idx}_y', f'{idx}_z', f'{idx}_v'])

with open(csv_save_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        row = [frame_idx]
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for idx in SELECTED_INDICES:
                lm = landmarks[idx]
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            # 시각화 (원본 프레임에 그리기)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for idx in SELECTED_INDICES:
                lm = landmarks[idx]
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        else:
            row.extend([0] * (len(SELECTED_INDICES) * 4))

        # 기록: CSV 파일 & 결과 영상 파일
        writer.writerow(row)
        out_video.write(frame) # 시각화된 프레임을 영상 파일로 저장

        # 화면 표시 (작동 확인용)
        cv2.imshow('Saving Result...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

cap.release()
out_video.release() # 저장 완료 후 닫기
cv2.destroyAllWindows()
print(f"✅ 저장 완료!\n1. 영상: {video_save_path}\n2. 데이터: {csv_save_path}")