"""
단일 동영상 keypoint 추출 + 스켈레톤 동영상 생성
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


def normalize_keypoints(keypoints: dict) -> dict:
    """체형 정규화 (어깨 너비 + 골반 중심)"""
    
    left_shoulder = np.array([keypoints['left_shoulder_x'], 
                             keypoints['left_shoulder_y']])
    right_shoulder = np.array([keypoints['right_shoulder_x'], 
                              keypoints['right_shoulder_y']])
    
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    
    if shoulder_width < 0.01:
        shoulder_width = 0.1
    
    left_hip = np.array([keypoints['left_hip_x'], 
                        keypoints['left_hip_y']])
    right_hip = np.array([keypoints['right_hip_x'], 
                         keypoints['right_hip_y']])
    
    hip_center = (left_hip + right_hip) / 2
    
    normalized = {}
    
    keypoint_names = [
        'nose', 
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    all_x = []
    all_y = []
    temp_normalized = {}
    
    for name in keypoint_names:
        x = keypoints[f'{name}_x']
        y = keypoints[f'{name}_y']
        z = keypoints[f'{name}_z']
        
        x_centered = x - hip_center[0]
        y_centered = y - hip_center[1]
        
        x_norm = x_centered / shoulder_width
        y_norm = y_centered / shoulder_width
        z_norm = z / shoulder_width
        
        temp_normalized[f'{name}_x'] = x_norm
        temp_normalized[f'{name}_y'] = y_norm
        temp_normalized[f'{name}_z'] = z_norm
        
        all_x.append(x_norm)
        all_y.append(y_norm)
        
        if f'{name}_visibility' in keypoints:
            temp_normalized[f'{name}_visibility'] = keypoints[f'{name}_visibility']
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_max = max(range_x, range_y)
    
    if range_max < 0.01:
        range_max = 1.0
    
    for name in keypoint_names:
        x_norm = temp_normalized[f'{name}_x']
        y_norm = temp_normalized[f'{name}_y']
        z_norm = temp_normalized[f'{name}_z']
        
        normalized[f'{name}_x'] = float((x_norm - min_x) / range_max)
        normalized[f'{name}_y'] = float((y_norm - min_y) / range_max)
        normalized[f'{name}_z'] = float(z_norm)
        
        if f'{name}_visibility' in temp_normalized:
            normalized[f'{name}_visibility'] = temp_normalized[f'{name}_visibility']
    
    return normalized


def draw_skeleton_on_black(keypoints: dict, img_width: int = 480, 
                           img_height: int = 640, 
                           color: tuple = (0, 255, 0)) -> np.ndarray:
    """검정 배경에 스켈레톤 그리기"""
    
    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    connections = [
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('right_wrist', 'right_pinky'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'),
        ('left_ankle', 'left_foot_index'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot_index'),
    ]
    
    # 연결선
    for start_name, end_name in connections:
        start_x = keypoints.get(f'{start_name}_x')
        start_y = keypoints.get(f'{start_name}_y')
        end_x = keypoints.get(f'{end_name}_x')
        end_y = keypoints.get(f'{end_name}_y')
        
        if start_x is None or end_x is None:
            continue
        
        x1 = int(start_x * img_width)
        y1 = int(start_y * img_height)
        x2 = int(end_x * img_width)
        y2 = int(end_y * img_height)
        
        cv2.line(canvas, (x1, y1), (x2, y2), color, 3)
    
    # 관절점
    keypoint_names = [
        'nose', 
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    for name in keypoint_names:
        x = keypoints.get(f'{name}_x')
        y = keypoints.get(f'{name}_y')
        
        if x is None:
            continue
        
        px = int(x * img_width)
        py = int(y * img_height)
        
        cv2.circle(canvas, (px, py), 6, color, -1)
    
    return canvas


def process_single_video(video_path: str, output_dir: str, video_name: str = "output"):
    """
    동영상 1개 처리
    
    Args:
        video_path: 입력 동영상 경로
        output_dir: 출력 폴더
        video_name: 출력 파일 이름
    
    Outputs:
        {output_dir}/{video_name}_keypoints.csv
        {output_dir}/{video_name}_skeleton.mp4
    """
    
    print("\n" + "=" * 60)
    print("🎬 단일 동영상 Keypoint 추출 & 스켈레톤 생성")
    print("=" * 60)
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 동영상 열기 실패: {video_path}")
        return
    
    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    
    # FPS 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    
    # 출력 설정
    frame_width = 480
    frame_height = 640
    
    csv_path = os.path.join(output_dir, f"{video_name}_keypoints.csv")
    video_output_path = os.path.join(output_dir, f"{video_name}_skeleton.mp4")
    
    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, 
                         (frame_width, frame_height))
    
    # 데이터 저장
    all_data = []
    frame_id = 0
    
    print(f"\n[1/2] Keypoint 추출 중...")
    print(f"   입력: {os.path.basename(video_path)}")
    print(f"   FPS: {fps:.2f}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # RGB 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose estimation
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Keypoint 추출
            keypoints = {}
            
            keypoint_indices = {
                0: 'nose',
                11: 'left_shoulder', 12: 'right_shoulder',
                13: 'left_elbow', 14: 'right_elbow',
                15: 'left_wrist', 16: 'right_wrist',
                17: 'left_pinky', 18: 'right_pinky',
                23: 'left_hip', 24: 'right_hip',
                25: 'left_knee', 26: 'right_knee',
                27: 'left_ankle', 28: 'right_ankle',
                29: 'left_heel', 30: 'right_heel',
                31: 'left_foot_index', 32: 'right_foot_index'
            }
            
            for idx, name in keypoint_indices.items():
                lm = landmarks[idx]
                keypoints[f'{name}_x'] = lm.x
                keypoints[f'{name}_y'] = lm.y
                keypoints[f'{name}_z'] = lm.z
                keypoints[f'{name}_visibility'] = lm.visibility
            
            # 정규화
            normalized = normalize_keypoints(keypoints)
            
            # DataFrame용 데이터
            row = {'frame_id': frame_id, 'timestamp': frame_id / fps}
            row.update(normalized)
            all_data.append(row)
            
            # 스켈레톤 그리기
            skeleton = draw_skeleton_on_black(normalized, frame_width, frame_height, 
                                             color=(0, 255, 0))
            
            # 프레임 번호 표시
            cv2.putText(skeleton, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(skeleton, f"Time: {frame_id/fps:.2f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # 동영상에 쓰기
            out.write(skeleton)
        
        frame_id += 1
        
        # 진행 상황
        if frame_id % 10 == 0:
            print(f"   처리 중... {frame_id}프레임")
    
    # 정리
    cap.release()
    out.release()
    pose.close()
    
    print(f"\n   ✅ 총 {frame_id}개 프레임 처리 완료")
    
    # CSV 저장
    print(f"\n[2/2] CSV 저장 중...")
    
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"   ✅ CSV 저장: {csv_path}")
    print(f"   ✅ 스켈레톤 동영상: {video_output_path}")
    
    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)
    print(f"\n출력 파일:")
    print(f"  1. {csv_path}")
    print(f"  2. {video_output_path}")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("단일 동영상 처리")
    print("=" * 60)
    
    # 입력
    print("\n💡 동영상 경로를 입력하세요:")
    video_path = input("동영상: ").strip().strip('"').strip("'")
    
    if not os.path.exists(video_path):
        print(f"❌ 파일 없음: {video_path}")
        exit()
    
    # 출력 폴더
    print("\n💡 출력 폴더를 입력하세요:")
    default_output_dir = os.path.dirname(video_path)
    print(f"   기본: {default_output_dir}")
    
    output_dir = input("출력 폴더 (Enter=기본): ").strip().strip('"').strip("'")
    
    if not output_dir:
        output_dir = default_output_dir
    
    # 파일명
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 실행
    process_single_video(video_path, output_dir, video_name)