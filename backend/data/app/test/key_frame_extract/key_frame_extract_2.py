import cv2
import mediapipe as mp
import numpy as np
import os

# --- 경로 설정 ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"      # 원본 영상 파일
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_2.mp4"       # 결과 영상 저장 경로
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images"   # 키프레임 이미지 저장 폴더

# 폴더가 없으면 생성
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- MediaPipe 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_badminton_swing(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    print(f"영상 분석 시작 (FPS: {fps})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # MediaPipe 분석
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        data = {'idx': frame_idx, 'img': frame.copy()}
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 필요 관절 좌표 (x, y)
            data['wrist'] = [lm[16].x, lm[16].y]  # 오른손목
            data['nose'] = [lm[0].x, lm[0].y]     # 코
            data['shoulder'] = [lm[12].x, lm[12].y] # 오른어깨
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    # --- 키프레임 추출 알고리즘 ---
    
    # 1. Impact: 손목 y값이 최소(가장 높음)인 프레임
    valid_frames = [f for f in frames if 'wrist' in f]
    impact_frame_data = min(valid_frames, key=lambda x: x['wrist'][1])
    i_idx = impact_frame_data['idx']

    # 2. Backswing: 임팩트 이전 프레임 중 손목이 코 높이와 비슷하고 머리 뒤에 있는 시점
    # (오른손잡이가 오른쪽을 보고 있다면 wrist_x < nose_x 가 뒤쪽임)
    backswing_candidates = []
    for i in range(0, i_idx):
        f = frames[i]
        if 'wrist' in f:
            # 손목이 코보다 뒤에 있는지 확인 (x좌표 비교 - 방향에 따라 유동적일 수 있음)
            # 여기서는 손목 y와 코 y의 차이가 최소인 지점을 우선 찾음
            y_diff = abs(f['wrist'][1] - f['nose'][1])
            backswing_candidates.append((i, y_diff))
    
    b_idx = min(backswing_candidates, key=lambda x: x[1])[0] if backswing_candidates else i_idx // 2

    # 3. Ready: 백스윙 이전 프레임 중 움직임(손목 y 변화)이 가장 적은 시점
    ready_candidates = []
    for i in range(1, b_idx):
        if 'wrist' in frames[i] and 'wrist' in frames[i-1]:
            movement = abs(frames[i]['wrist'][1] - frames[i-1]['wrist'][1])
            ready_candidates.append((i, movement))
    
    r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0

    return frames, (r_idx, b_idx, i_idx), fps

# --- 실행 및 저장 ---
all_frames, (r_idx, b_idx, i_idx), original_fps = analyze_badminton_swing(INPUT_VIDEO)

print(f"추출 완료: READY({r_idx}), BACKSWING({b_idx}), IMPACT({i_idx})")

# 1. 결과 영상 및 이미지 저장
height, width, _ = all_frames[0]['img'].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, original_fps, (width, height))

keyframe_map = {r_idx: "READY", b_idx: "BACKSWING", i_idx: "IMPACT"}

for f in all_frames:
    img = f['img']
    if f['idx'] in keyframe_map:
        label = keyframe_map[f['idx']]
        color = (0, 255, 0) if label == "READY" else (255, 255, 0) if label == "BACKSWING" else (0, 0, 255)
        
        # 영상에 텍스트 표시
        cv2.putText(img, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # 이미지 파일로 추출
        img_filename = os.path.join(IMAGE_SAVE_DIR, f"{label}.jpg")
        cv2.imwrite(img_filename, img)
        print(f"이미지 저장됨: {img_filename}")

    out.write(img)

out.release()
print(f"\n모든 작업이 완료되었습니다.")
print(f"- 영상 저장: {OUTPUT_VIDEO}")
print(f"- 이미지 폴더: {IMAGE_SAVE_DIR}")