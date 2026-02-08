import cv2
import mediapipe as mp
import csv
import os

# --- 1. 경로 및 포인트 설정 ---
VIDEO_PATH = '/Users/minji/Documents/pro_swing_3_GT.mp4'  # 영상 경로 (직접 수정)
SAVE_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_3_data.csv' # 저장 경로 (직접 수정)

# 민지가 선택한 19개의 핵심 포인트 (오름차순 정렬)
SELECTED_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. 영상 로드 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"분석 시작: {VIDEO_PATH} (FPS: {fps:.2f})")

# --- 3. CSV 준비 (선택된 번호만 헤더 생성) ---
header = ['frame']
for idx in SELECTED_INDICES:
    header.extend([f'{idx}_x', f'{idx}_y', f'{idx}_z', f'{idx}_v'])

with open(SAVE_PATH, mode='w', newline='') as f:
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
            # 전체 33개 중 SELECTED_INDICES에 해당하는 것만 추출
            landmarks = results.pose_landmarks.landmark
            for idx in SELECTED_INDICES:
                lm = landmarks[idx]
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            # 인식 실패 시 해당 프레임은 0으로 채움
            row.extend([0] * (len(SELECTED_INDICES) * 4))

        writer.writerow(row)
        
        if frame_idx % 30 == 0:
            print(f"진행 중: {frame_idx} 프레임 처리 완료")
        frame_idx += 1

cap.release()
pose.close()
print(f"🎉 추출 완료! 저장 위치: {os.path.abspath(SAVE_PATH)}")