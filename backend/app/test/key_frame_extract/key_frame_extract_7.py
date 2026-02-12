import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- 경로 설정 (사용자님 원본 경로로 고정) ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_7.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_7"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- MediaPipe 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산 (b가 손목/꼭짓점)"""
    a = np.array(a) # Elbow (14)
    b = np.array(b) # Wrist (16)
    c = np.array(c) # Hand (17, 18 중점)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

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
            # 관절 데이터 추출
            data['wrist'] = [lm[16].x, lm[16].y]    # 오른손목
            data['elbow'] = [lm[14].x, lm[14].y]    # 오른팔꿈치
            data['nose'] = [lm[0].x, lm[0].y]       # 코
            data['shoulder'] = [lm[12].x, lm[12].y] # 오른어깨
            # 손(Hand) 좌표: 17번과 18번의 중점
            data['hand'] = [(lm[17].x + lm[18].x)/2, (lm[17].y + lm[18].y)/2]
            
            # 손목 스냅 각도 계산
            data['snap_angle'] = calculate_angle(data['elbow'], data['wrist'], data['hand'])
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'snap_angle' in f]
    
    # -----------------------------------------------------------------
    # 1. IMPACT (수정됨): 손목 최고점(y 최소) 이후, 각도가 160.5도와 가장 근사한 지점
    # -----------------------------------------------------------------
    # 손목이 가장 높이 올라간 프레임 찾기
    highest_wrist_frame = min(valid_frames, key=lambda x: x['wrist'][1])
    highest_idx = highest_wrist_frame['idx']
    
    # 최고점 이후 프레임들 필터링
    after_highest_frames = [f for f in valid_frames if f['idx'] >= highest_idx]
    
    target_angle = 160.5
    if after_highest_frames:
        impact_frame_data = min(after_highest_frames, key=lambda x: abs(x['snap_angle'] - target_angle))
        i_idx = impact_frame_data['idx']
    else:
        i_idx = highest_idx # 이후 프레임이 없을 경우 최고점을 임팩트로 사용

    # -----------------------------------------------------------------
    # 2. BACKSWING (유지): 임팩트 이전 구간 중 |wrist_x - elbow_x| 가 최소일 때
    # -----------------------------------------------------------------
    backswing_candidates = []
    for f in valid_frames:
        if f['idx'] < i_idx: 
            x_diff = abs(f['wrist'][0] - f['elbow'][0])
            backswing_candidates.append((f['idx'], x_diff))
    
    if backswing_candidates:
        b_idx = min(backswing_candidates, key=lambda x: x[1])[0]
    else:
        b_idx = i_idx // 2

    # -----------------------------------------------------------------
    # 3. READY (유지): 백스윙 이전 구간 중 손목 y축 움직임이 가장 적을 때
    # -----------------------------------------------------------------
    ready_candidates = []
    for i in range(1, b_idx):
        if 'wrist' in frames[i] and 'wrist' in frames[i-1]:
            movement = abs(frames[i]['wrist'][1] - frames[i-1]['wrist'][1])
            ready_candidates.append((i, movement))
    
    r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0

    return frames, (r_idx, b_idx, i_idx), fps

# --- 실행 및 저장 부분 ---
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
        
        # 영상에 라벨 표시
        cv2.putText(img, f"{label} (F:{curr_idx})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # 이미지 파일 저장 (동일 이름 시 덮어쓰기됨)
        img_filename = os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{curr_idx}.jpg")
        cv2.imwrite(img_filename, img)
        print(f"이미지 저장됨: {img_filename}")

    out.write(img)

out.release()
print(f"\n작업 완료! {IMAGE_SAVE_DIR} 폴더를 확인해 보세요.")