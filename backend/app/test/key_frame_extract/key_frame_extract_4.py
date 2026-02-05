import cv2
import mediapipe as mp
import numpy as np
import os

# --- 경로 설정 ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_4.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_4"

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

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        data = {'idx': frame_idx, 'img': frame.copy()}
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 분석에 필요한 관절 데이터 추가 추출 (팔꿈치 추가)
            data['wrist'] = [lm[16].x, lm[16].y]    # 오른손목
            data['elbow'] = [lm[14].x, lm[14].y]    # 오른팔꿈치
            data['nose'] = [lm[0].x, lm[0].y]       # 코
            data['shoulder'] = [lm[12].x, lm[12].y] # 오른어깨
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'nose' in f]
    
    # 1. IMPACT 추출 (기존 유지: 손목 y값이 최소일 때)
    impact_frame_data = min(valid_frames, key=lambda x: x['wrist'][1])
    i_idx = impact_frame_data['idx']

    # 2. BACKSWING 추출 (수정됨)
    # 조건: x축 기준 손목 < 팔꿈치 < 코 순서이면서 팔꿈치가 가장 높이 올라간 시점
    backswing_candidates = []
    for f in valid_frames:
        if f['idx'] < i_idx: # 임팩트 이전이어야 함
            w_x, e_x, n_x = f['wrist'][0], f['elbow'][0], f['nose'][0]
            
            # 사용자 요청 조건: 왼쪽부터 손목-팔꿈치-코 순서 (x좌표가 작을수록 왼쪽)
            if w_x < e_x < n_x:
                # 후보들 중 팔꿈치 높이(y)를 기준으로 선정하기 위해 y값 저장
                backswing_candidates.append((f['idx'], f['elbow'][1]))
    
    # 조건을 만족하는 후보 중 팔꿈치가 가장 높은(y가 가장 작은) 프레임 선택
    if backswing_candidates:
        b_idx = min(backswing_candidates, key=lambda x: x[1])[0]
    else:
        # 조건을 만족하는 프레임이 없을 경우를 대비한 예외 처리 (임팩트 절반 지점)
        b_idx = i_idx // 2

    # 3. READY 추출 (기존 유지: 백스윙 이전 중 움직임이 가장 적을 때)
    ready_candidates = []
    for i in range(1, b_idx):
        if 'wrist' in frames[i] and 'wrist' in frames[i-1]:
            movement = abs(frames[i]['wrist'][1] - frames[i-1]['wrist'][1])
            ready_candidates.append((i, movement))
    
    r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0

    return frames, (r_idx, b_idx, i_idx), fps

# --- 실행 및 저장 부분 (기존 동일) ---
all_frames, (r_idx, b_idx, i_idx), original_fps = analyze_badminton_swing(INPUT_VIDEO)
print(f"추출 완료: READY({r_idx}), BACKSWING({b_idx}), IMPACT({i_idx})")

height, width, _ = all_frames[0]['img'].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, original_fps, (width, height))

keyframe_map = {r_idx: "READY", b_idx: "BACKSWING", i_idx: "IMPACT"}

for f in all_frames:
    img = f['img']
    curr_idx = f['idx']
    
    if curr_idx in keyframe_map:
        label = keyframe_map[curr_idx]
        color = (0, 255, 0) if label == "READY" else (255, 255, 0) if label == "BACKSWING" else (0, 0, 255)
        cv2.putText(img, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        img_filename = os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{curr_idx}.jpg")
        cv2.imwrite(img_filename, img)

    out.write(img)

out.release()
print(f"\n작업 완료! {IMAGE_SAVE_DIR} 폴더를 확인해 보세요.")