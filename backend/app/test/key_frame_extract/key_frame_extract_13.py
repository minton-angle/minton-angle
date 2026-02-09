import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- 경로 설정 (사용자님 환경 유지) ---
INPUT_VIDEO = "/Users/minji/Documents/Baekcoach.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_Baek_2.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_Baek_2"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- MediaPipe 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

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
            data['nose'] = [lm[0].x, lm[0].y]
            data['shoulder'] = [lm[12].x, lm[12].y] 
            data['elbow'] = [lm[14].x, lm[14].y]
            data['wrist'] = [lm[16].x, lm[16].y]
            data['hand'] = [(lm[17].x + lm[18].x)/2, (lm[17].y + lm[18].y)/2]
            data['snap_angle'] = calculate_angle(data['elbow'], data['wrist'], data['hand'])
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'shoulder' in f]

    # 0. 기준점: 손목 최고점
    highest_wrist_frame = min(valid_frames, key=lambda x: x['wrist'][1])
    highest_idx = highest_wrist_frame['idx']

    # -----------------------------------------------------------------
    # 2. READY (수정됨): 팔꿈치가 가장 왼쪽 + 정적 상태(대기 시간)
    # -----------------------------------------------------------------
    ready_candidates = []
    # 최고점 이전 구간에서 탐색
    search_range = [f for f in valid_frames if f['idx'] < highest_idx]
    
    if search_range:
        # 팔꿈치 x좌표의 최솟값(가장 왼쪽) 찾기
        min_elbow_x = min(search_range, key=lambda x: x['elbow'][0])['elbow'][0]
        
        for f in search_range:
            i = f['idx']
            if i > 0 and 'wrist' in frames[i-1]:
                # 1. 팔꿈치가 왼쪽 끝 지점 부근에 있는지 (오차범위 0.05 이내)
                is_leftmost = abs(f['elbow'][0] - min_elbow_x) < 0.05
                # 2. 움직임(대기 시간 지표) 계산
                movement = abs(f['wrist'][1] - frames[i-1]['wrist'][1])
                
                if is_leftmost:
                    ready_candidates.append((i, movement))
        
        # 조건을 만족하는 후보 중 움직임이 가장 적은(정지한) 프레임 선택
        r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0
    else:
        r_idx = 0

    # 1. IMPACT (기존 유지)
    impact_candidates = [f for f in valid_frames if f['idx'] >= highest_idx and f['wrist'][1] <= f['elbow'][1]]
    i_idx = min(impact_candidates, key=lambda x: abs(x['snap_angle'] - 160.5))['idx'] if impact_candidates else highest_idx

    # 3. BACKSWING (기존 유지)
    bs_candidates = []
    for f in valid_frames:
        if r_idx < f['idx'] <= highest_idx:
            is_behind = f['wrist'][0] < f['nose'][0]
            is_lifted = f['elbow'][1] < f['shoulder'][1] + 0.05 
            if is_behind and is_lifted:
                x_diff = abs(f['wrist'][0] - f['elbow'][0])
                bs_candidates.append((f['idx'], x_diff))

    if bs_candidates:
        min_diff = min(bs_candidates, key=lambda x: x[1])[1]
        b_idx = next(idx for idx, diff in bs_candidates if diff <= min_diff * 1.1)
    else:
        b_idx = min([f for f in valid_frames if r_idx < f['idx'] <= highest_idx], 
                    key=lambda f: abs(f['wrist'][0] - f['elbow'][0]))['idx']

    return frames, (r_idx, b_idx, i_idx), fps

# --- 결과 실행 및 저장 ---
all_frames, (r_idx, b_idx, i_idx), original_fps = analyze_badminton_swing(INPUT_VIDEO)
print(f"최종 추출 결과: READY({r_idx}), BACKSWING({b_idx}), IMPACT({i_idx})")

height, width, _ = all_frames[0]['img'].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, original_fps, (width, height))

keyframe_map = {r_idx: "READY", b_idx: "BACKSWING", i_idx: "IMPACT"}

for f in all_frames:
    img = f['img']
    if f['idx'] in keyframe_map:
        label = keyframe_map[f['idx']]
        color = (0, 255, 0) if label == "READY" else (255, 255, 0) if label == "BACKSWING" else (0, 0, 255)
        cv2.putText(img, f"{label} (F:{f['idx']})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{f['idx']}.jpg"), img)
    out.write(img)

out.release()
print(f"\n작업 완료! {IMAGE_SAVE_DIR} 폴더를 확인해 보세요.")