import cv2
import mediapipe as mp
import numpy as np
import os

# --- 설정 (경로를 수정하세요) ---
INPUT_VIDEO = "/Users/minji/Documents/pro_swing_1_GT.mp4"      # 원본 영상 파일 경로
OUTPUT_VIDEO = "/Users/minji/Documents/minton-angle/backend/data/standard/analyzed_swing_3.mp4"     # 결과 영상 저장 경로
IMAGE_SAVE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/keyframes_images_3"        # 키프레임 이미지 저장 폴더
# ----------------------------

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_swing_and_save(video_path, output_video_path, image_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_data = []
    frame_idx = 0

    print(f"영상 분석 시작 (Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, FPS: {fps})")

    # 1. 전체 프레임 데이터 분석 및 저장
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_info = {'idx': frame_idx, 'img': frame}
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 필요한 관절 좌표 추출 (코, 오른어깨, 오른팔꿈치, 오른손목)
            frame_info['nose'] = (lm[mp_pose.PoseLandmark.NOSE].x * frame_width, lm[mp_pose.PoseLandmark.NOSE].y * frame_height)
            frame_info['shoulder'] = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
            frame_info['elbow'] = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame_width, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame_height)
            frame_info['wrist'] = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height)
        
        frames_data.append(frame_info)
        frame_idx += 1
    cap.release()
    
    if not frames_data or 'wrist' not in frames_data[0]:
         print("Error: No pose data detected in video.")
         return

    # --- 키프레임 추출 알고리즘 ---
    valid_frames = [f for f in frames_data if 'wrist' in f]

    # 1. Impact (임팩트): 손목이 가장 높은(y값 최소) 순간
    impact_frame = min(valid_frames, key=lambda f: f['wrist'][1])
    i_idx = impact_frame['idx']

    # 2. Ready (준비): 임팩트 이전, 손목 y 변화가 가장 적은 정적 구간
    ready_candidates = []
    for i in range(1, min(i_idx, len(valid_frames))):
        curr = valid_frames[i]
        prev = valid_frames[i-1]
        if curr['idx'] < i_idx:
            # 어깨보다 손목이 낮거나 비슷할 때만 준비 자세 후보로 고려
            if curr['wrist'][1] >= curr['shoulder'][1]:
                movement = abs(curr['wrist'][1] - prev['wrist'][1])
                ready_candidates.append((curr['idx'], movement))
    
    # 움직임이 가장 적은 프레임 선택, 후보가 없으면 시작 프레임
    r_idx = min(ready_candidates, key=lambda x: x[1])[0] if ready_candidates else valid_frames[0]['idx']

    # 3. Backswing (백스윙) - 사용자 요청 로직 적용
    # 조건: Ready와 Impact 사이에서, 팔꿈치가 코보다 높고(y값 작음), 손목-팔꿈치-코 순으로 뒤에 위치
    backswing_candidates = []
    for i in range(0, len(valid_frames)):
        f = valid_frames[i]
        if r_idx < f['idx'] < i_idx:
            # 조건 검사 (오른쪽 보고 서 있는 자세 기준: x값은 오른쪽이 큼)
            # 1. 팔꿈치 높이가 코보다 높아야 함 (y값 비교)
            cond_height = f['elbow'][1] < f['nose'][1]
            # 2. 앞뒤 순서가 손목 < 팔꿈치 < 코 여야 함 (x값 비교)
            cond_order = (f['wrist'][0] < f['elbow'][0]) and (f['elbow'][0] < f['nose'][0])
            
            if cond_height and cond_order:
                backswing_candidates.append(f)

    # 후보 중 팔꿈치가 가장 높이 올라간 순간 선택, 없으면 중간 지점 대체
    if backswing_candidates:
        backswing_frame = min(backswing_candidates, key=lambda f: f['elbow'][1])
        b_idx = backswing_frame['idx']
    else:
        print("Warning: 이상적인 백스윙 조건을 만족하는 프레임이 없습니다. 중간 지점으로 대체합니다.")
        b_idx = (r_idx + i_idx) // 2

    print(f"추출 결과: READY({r_idx}) -> BACKSWING({b_idx}) -> IMPACT({i_idx})")

    # --- 결과 저장 (영상 및 이미지) ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    keyframe_indices = {r_idx: "READY", b_idx: "BACKSWING", i_idx: "IMPACT"}
    saved_counts = 0

    for f_data in frames_data:
        img = f_data['img'].copy()
        idx = f_data['idx']
        
        if idx in keyframe_indices:
            label = keyframe_indices[idx]
            color = (0, 255, 0) if label == "READY" else (255, 255, 0) if label == "BACKSWING" else (0, 0, 255)
            
            # 라벨 표시
            cv2.putText(img, f"{label} ({idx})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            if 'wrist' in f_data:
                 cv2.circle(img, (int(f_data['wrist'][0]), int(f_data['wrist'][1])), 15, color, -1)

            # 이미지 파일 저장 (파일명에 프레임 번호 포함)
            img_filename = os.path.join(image_dir, f"{label}_frame_{idx}.jpg")
            cv2.imwrite(img_filename, img)
            print(f"이미지 저장: {img_filename}")
            saved_counts += 1
            
        out.write(img)
        
    out.release()
    print(f"\n완료! 총 {saved_counts}개의 키프레임 이미지가 '{image_dir}'에 저장되었습니다.")
    print(f"결과 영상: {output_video_path}")

# 실행
analyze_swing_and_save(INPUT_VIDEO, OUTPUT_VIDEO, IMAGE_SAVE_DIR)