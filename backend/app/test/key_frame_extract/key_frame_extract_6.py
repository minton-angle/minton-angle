import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- 경로 설정 (사용자님 환경 유지) ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_6.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_6"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- MediaPipe 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산 (b가 꼭짓점/손목)"""
    a = np.array(a) # Elbow (14)
    b = np.array(b) # Wrist (16)
    c = np.array(c) # Hand (17, 18의 중점)
    
    ba = a - b
    bc = c - b
    
    # 코사인 유사도를 이용한 각도 계산
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
            
            # 손 부분: 17번(새끼손가락 안쪽)과 18번(새끼손가락 바깥쪽)의 중앙값 사용
            hand_x = (lm[17].x + lm[18].x) / 2
            hand_y = (lm[17].y + lm[18].y) / 2
            data['hand'] = [hand_x, hand_y]
            
            # 현재 프레임의 손목 스냅 각도 계산 (팔꿈치-손목-손)
            data['snap_angle'] = calculate_angle(data['elbow'], data['wrist'], data['hand'])
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    # 데이터가 유효한 프레임만 필터링
    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'snap_angle' in f]
    
    # -----------------------------------------------------------------
    # 1. IMPACT (수정됨): 각도가 160.5도와 가장 근사한 지점 탐색
    # -----------------------------------------------------------------
    target_angle = 160.5
    impact_frame_data = min(valid_frames, key=lambda x: abs(x['snap_angle'] - target_angle))
    i_idx = impact_frame_data['idx']

    # 2. BACKSWING (기존 유지): 손목과 팔꿈치의 x좌표가 가장 비슷해지는 시점 (임팩트 이전)
    backswing_candidates = []
    for f in valid_frames:
        if f['idx'] < i_idx: # 임팩트 이전 구간에서 탐색
            x_diff = abs(f['wrist'][0] - f['elbow'][0])
            backswing_candidates.append((f['idx'], x_diff))
    
    if backswing_candidates:
        b_idx = min(backswing_candidates, key=lambda x: x[1])[0]
    else:
        b_idx = i_idx // 2

    # 3. READY (기존 유지): 백스윙 이전 중 손목 움직임이 가장 적을 때
    ready_candidates = []
    for i in range(1, b_idx):
        if 'wrist' in frames[i] and 'wrist' in frames[i-1]:
            movement = abs(frames[i]['wrist'][1] - frames[i-1]['wrist'][1])
            ready_candidates.append((i, movement))
    
    r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0

    return frames, (r_idx, b_idx, i_idx), fps

# --- 실행 및 저장 부분 (사용자님 코드 유지) ---
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
        
        # 이미지 파일 저장
        img_filename = os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{curr_idx}.jpg")
        cv2.imwrite(img_filename, img)
        print(f"이미지 저장됨: {img_filename}")

    out.write(img)

out.release()
print(f"\n모든 작업이 완료되었습니다! {IMAGE_SAVE_DIR} 폴더를 확인해 보세요.")