import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- 경로 설정 ---
INPUT_VIDEO = "/Users/minji/Documents/Baekcoach.mp4"
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_Baek_3.mp4"
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_Baek_3"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# --- MediaPipe 설정 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_dist(p1, p2):
    """두 점 사이의 유클리드 거리를 계산"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

    valid_frames = [f for f in frames if 'wrist' in f and 'elbow' in f and 'nose' in f]

    # 0. 기준점: 손목 최고점
    highest_wrist_frame = min(valid_frames, key=lambda x: x['wrist'][1])
    highest_idx = highest_wrist_frame['idx']

    # -----------------------------------------------------------------
    # 1. READY: 팔꿈치가 가장 왼쪽 + 정적 상태 (유지)
    # -----------------------------------------------------------------
    ready_candidates = []
    search_range_ready = [f for f in valid_frames if f['idx'] < highest_idx]
    if search_range_ready:
        min_elbow_x = min(search_range_ready, key=lambda x: x['elbow'][0])['elbow'][0]
        for f in search_range_ready:
            i = f['idx']
            if i > 0 and 'wrist' in frames[i-1]:
                is_leftmost = abs(f['elbow'][0] - min_elbow_x) < 0.05
                movement = abs(f['wrist'][1] - frames[i-1]['wrist'][1])
                if is_leftmost: ready_candidates.append((i, movement))
        r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0
    else: r_idx = 0

    # -----------------------------------------------------------------
    # 2. BACKSWING (최종 수정): READY 이후 ~ 최고점 사이
    #    '손목이 코보다 왼쪽' + '코, 팔꿈치, 손목 사이의 거리 합 최소'
    # -----------------------------------------------------------------
    bs_candidates = []
    for f in valid_frames:
        # 범위: READY 이후 ~ 손목 최고점 이전
        if r_idx < f['idx'] <= highest_idx:
            # 조건: 손목이 코보다 왼쪽 (화면 좌표계 기준 x가 작음)
            if f['wrist'][0] < f['nose'][0]:
                # 코, 팔꿈치, 손목 세 점 사이의 거리 합 계산 (삼각형 둘레)
                d1 = get_dist(f['nose'], f['elbow'])
                d2 = get_dist(f['elbow'], f['wrist'])
                d3 = get_dist(f['wrist'], f['nose'])
                total_dist = d1 + d2 + d3
                bs_candidates.append((f['idx'], total_dist))

    if bs_candidates:
        # 세 점 사이의 거리가 가장 가까운(최소인) 프레임 선택
        b_idx = min(bs_candidates, key=lambda x: x[1])[0]
    else:
        # 조건을 만족하는 프레임이 없을 경우 중간값 사용
        b_idx = (r_idx + highest_idx) // 2

    # -----------------------------------------------------------------
    # 3. IMPACT: 최고점 이후 구간 내 160.5도 근사 (유지)
    # -----------------------------------------------------------------
    impact_candidates = [f for f in valid_frames if f['idx'] >= highest_idx and f['wrist'][1] <= f['elbow'][1]]
    i_idx = min(impact_candidates, key=lambda x: abs(x['snap_angle'] - 160.5))['idx'] if impact_candidates else highest_idx

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
    if f['idx'] in keyframe_map:
        label = keyframe_map[f['idx']]
        color = (0, 255, 0) if label == "READY" else (255, 255, 0) if label == "BACKSWING" else (0, 0, 255)
        cv2.putText(img, f"{label} (F:{f['idx']})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, f"{label}_frame_{f['idx']}.jpg"), img)
    out.write(img)

out.release()
print(f"\n작업 완료! {IMAGE_SAVE_DIR} 폴더를 확인해 보세요.")