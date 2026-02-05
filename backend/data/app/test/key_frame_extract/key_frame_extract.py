import cv2
import mediapipe as mp
import numpy as np
import math

# --- 설정 (경로를 수정하세요) ---
INPUT_VIDEO_PATH = "/Users/minji/Documents/pro_swing_1_GT.mp4"      # 분석할 영상 경로
OUTPUT_VIDEO_PATH = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing.mp4" # 저장될 영상 경로
# ----------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산 (b가 중심)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_keyframes(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"영상 FPS: {fps}")
    
    frame_data = []
    frame_idx = 0
    
    print("관절 데이터 분석 중...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 필요 관절: 코(0), 오른어깨(12), 오른팔꿈치(14), 오른손목(16)
            nose = [lm[0].x, lm[0].y]
            shoulder = [lm[12].x, lm[12].y]
            elbow = [lm[14].x, lm[14].y]
            wrist = [lm[16].x, lm[16].y]
            
            # 팔꿈치 각도 계산
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            
            frame_data.append({
                'idx': frame_idx,
                'wrist_y': wrist[1],
                'wrist_pos': wrist,
                'shoulder_y': shoulder[1],
                'nose_pos': nose,
                'elbow_angle': elbow_angle
            })
        frame_idx += 1
    cap.release()

    # --- 키프레임 판단 로직 ---
    
    # 1. 임팩트(Impact) 찾기: 손목의 y좌표 변화량(속도)이 급격히 커지는 지점
    velocities = []
    for i in range(1, len(frame_data)):
        v = abs(frame_data[i]['wrist_y'] - frame_data[i-1]['wrist_y'])
        velocities.append(v)
    
    # 가속도가 가장 붙는 시점 (손목이 위로 솟구치는 순간)을 임팩트로 추정
    impact_idx = np.argmax(velocities) + 1
    
    # 2. 준비(Ready) 찾기: 임팩트 이전, 손목이 머리 부근에 있고 속도가 최소(정지)인 지점
    ready_candidates = []
    for i in range(0, impact_idx):
        d = frame_data[i]
        # 손목이 어깨보다 뒤(y값이 작거나 비슷)이고 코 높이 부근일 때
        if d['wrist_y'] < d['shoulder_y'] + 0.1: 
            vel = velocities[i-1] if i > 0 else 1
            ready_candidates.append((i, vel))
            
    # 속도가 가장 0에 가까운 지점 선택
    ready_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else 0
    
    # 3. 백스윙(Backswing) 찾기: 준비와 임팩트 사이에서 팔꿈치 각도가 최소인 지점
    swing_range = frame_data[ready_idx : impact_idx]
    backswing_idx = ready_idx + min(range(len(swing_range)), key=lambda i: swing_range[i]['elbow_angle'])

    return (ready_idx, backswing_idx, impact_idx), fps

# --- 메인 실행 및 저장 ---
keyframes, video_fps = extract_keyframes(INPUT_VIDEO_PATH)
r_idx, b_idx, i_idx = keyframes

print(f"\n추출된 키프레임: Ready({r_idx}), Backswing({b_idx}), Impact({i_idx})")

# 결과 영상 저장
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, 
                     (int(cap.get(3)), int(cap.get(4))))

f_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    label = ""
    color = (255, 255, 255)
    
    if f_count == r_idx: label, color = "READY", (0, 255, 0)
    elif f_count == b_idx: label, color = "BACKSWING", (255, 255, 0)
    elif f_count == i_idx: label, color = "IMPACT", (0, 0, 255)
    
    if label:
        cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.circle(frame, (50, 120), 20, color, -1)

    out.write(frame)
    f_count += 1

cap.release()
out.release()
print(f"분석 영상 저장 완료: {OUTPUT_VIDEO_PATH}")