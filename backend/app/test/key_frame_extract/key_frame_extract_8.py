import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- 경로 설정 (사용자님 환경 유지) ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_8.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_8"

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
            # 필요 관절 좌표 추출
            data['wrist'] = [lm[16].x, lm[16].y]    # 오른손목
            data['elbow'] = [lm[14].x, lm[14].y]    # 오른팔꿈치
            data['nose'] = [lm[0].x, lm[0].y]       # 코
            data['shoulder'] = [lm[12].x, lm[12].y] # 오른어깨
            data['hand'] = [(lm[17].x + lm[18].x)/2, (lm[17].y + lm[18].y)/2]
            
            # 손목 스냅 각도 계산 (팔꿈치-손목-손)
            data['snap_angle'] = calculate_angle(data['elbow'], data['wrist'], data['hand'])
        
        frames.append(data)
        frame_idx += 1
    cap.release()

    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'snap_angle' in f]
    
    # -----------------------------------------------------------------
    # 1. IMPACT (수정됨): 손목 최고점 ~ 손목이 팔꿈치보다 위에 있는 구간 내 탐색
    # -----------------------------------------------------------------
    # A. 손목 최고점(y 최소값) 프레임 찾기
    highest_wrist_frame = min(valid_frames, key=lambda x: x['wrist'][1])
    highest_idx = highest_wrist_frame['idx']
    
    # B. 구간 설정: 최고점 이후이면서 '손목이 팔꿈치보다 위에 있는(y_wrist < y_elbow)' 프레임들
    impact_candidates = []
    for f in valid_frames:
        # 손목 최고점 시점부터 탐색 시작
        if f['idx'] >= highest_idx:
            # 손목이 팔꿈치보다 위에 있는 조건 확인 (y좌표는 작을수록 위쪽임)
            if f['wrist'][1] < f['elbow'][1]:
                impact_candidates.append(f)
            else:
                # 손목이 팔꿈치보다 아래로 내려가는 순간 탐색 종료
                if impact_candidates: # 이미 후보가 들어있다면 루프 탈출
                    break

    # C. 해당 후보군 중 160.5도와 가장 가까운 프레임 추출
    target_angle = 160.5
    if impact_candidates:
        impact_frame_data = min(impact_candidates, key=lambda x: abs(x['snap_angle'] - target_angle))
        i_idx = impact_frame_data['idx']
    else:
        # 조건을 만족하는 구간이 짧거나 없을 경우 최고점을 임팩트로 사용
        i_idx = highest_idx

    # -----------------------------------------------------------------
    # 2. BACKSWING (유지): 임팩트 이전 중 |wrist_x - elbow_x| 가 최소일 때
    # -----------------------------------------------------------------
    backswing_candidates = []
    for f in valid_frames:
        if f['idx'] < i_idx:
            x_diff = abs(f['wrist'][0] - f['elbow'][0])
            backswing_candidates.append((f['idx'], x_diff))
    
    b_idx = min(backswing_candidates, key=lambda x: x[1])[0] if backswing_candidates else i_idx // 2

    # -----------------------------------------------------------------
    # 3. READY (유지): 백스윙 이전 중 손목 y축 움직임이 가장 적을 때
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
print(f"최종 추출 결과: READY({r_idx}), BACKSWING({b_idx}), IMPACT({i_idx})")

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
        cv2.putText(img, f"{label} (F:{curr_idx})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        img_filename = os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{curr_idx}.jpg")
        cv2.imwrite(img_filename, img)

    out.write(img)

out.release()
print(f"\n분석 완료! '{IMAGE_SAVE_DIR}' 폴더에서 이미지를 확인하세요.")