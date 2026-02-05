import cv2
import mediapipe as mp
import pandas as pd
import os

# 1. MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

# 2. 은서님이 최종 결정한 19개 관절 번호
TARGET_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def generate_gt_csv(video_filename, output_name):
    # --- [경로 수정 핵심 파트] ---
    # 현재 실행 중인 파일(gt_generator.py)의 절대 경로를 가져옵니다.
    current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/app/test
    
    # 프로젝트 루트(backend)로 올라가서 data/videos 폴더로 이동합니다.
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    input_path = os.path.join(project_root, "data", "videos", video_filename)
    output_dir = os.path.join(project_root, "data", "standard")
    output_path = os.path.join(output_dir, f"{output_name}.csv")
    # --------------------------

    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 폴더 생성됨: {output_dir}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 영상을 찾을 수 없습니다. 경로를 확인하세요:")
        print(f"   시도한 경로: {input_path}")
        return

    data_list = []
    print(f"🔄 분석 시작: {video_filename}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 분석을 위해 이미지 변환
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            row = {"frame": current_frame}
            
            for idx in TARGET_INDICES:
                row[f"{idx}_x"] = round(lm[idx].x, 6)
                row[f"{idx}_y"] = round(lm[idx].y, 6)
                row[f"{idx}_v"] = round(lm[idx].visibility, 6)
            
            data_list.append(row)
            
    if not data_list:
        print(f"⚠️ {video_filename}에서 관절 데이터를 추출하지 못했습니다.")
        return

    df = pd.DataFrame(data_list)
    df.to_csv(output_path, index=False)
    cap.release()
    print(f"✅ CSV 생성 완료: {output_path}")
    print(f"   (총 {len(df)}프레임 추출됨)")

if __name__ == "__main__":
    # 영상 파일명이 맞는지 다시 한번 확인해주세요! (standard_01.mp4 인지)
    generate_gt_csv("standard_01.mp4", "standard_01_gt")
    generate_gt_csv("standard_02.mp4", "standard_02_gt")