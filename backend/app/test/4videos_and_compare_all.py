"""
여러 GT 동영상 통합 처리
1. Keypoint 추출 및 정규화
2. 정규화된 CSV 저장
3. DTW 동기화 비교 동영상 생성
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


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


def extract_and_save_keypoints(video_path: str, output_csv: str) -> list:
    """
    동영상에서 keypoint 추출, 정규화, CSV 저장
    
    Returns:
        정규화된 keypoint 리스트
    """
    
    print(f"\n   📹 {os.path.basename(video_path)} 처리 중...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"      ❌ 동영상 열기 실패!")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"      총 {total_frames}프레임, {fps:.2f}fps")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    
    all_keypoints = []
    all_data = []
    
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
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
            all_keypoints.append(normalized)
            
            # CSV용 데이터
            row = {
                'frame_id': frame_id,
                'timestamp': frame_id / fps
            }
            row.update(normalized)
            all_data.append(row)
        else:
            all_keypoints.append({})
        
        frame_id += 1
        
        if frame_id % 10 == 0:
            progress = (frame_id / total_frames) * 100
            print(f"      진행: {progress:.1f}%", end='\r')
    
    cap.release()
    pose.close()
    
    # CSV 저장
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n      ✅ CSV 저장: {os.path.basename(output_csv)}")
    print(f"      {len(all_keypoints)}개 프레임")
    
    return all_keypoints


def keypoints_to_vector(keypoints: dict) -> np.ndarray:
    """Keypoint를 1차원 벡터로 변환 (DTW용)"""
    
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
    
    vector = []
    
    for name in keypoint_names:
        x = keypoints.get(f'{name}_x', 0.5)
        y = keypoints.get(f'{name}_y', 0.5)
        vector.extend([x, y])
    
    return np.array(vector)


def compute_dtw_alignment(ref_keypoints: list, target_keypoints: list) -> list:
    """DTW로 두 시퀀스 정렬"""
    
    print(f"      DTW 정렬 중... ({len(ref_keypoints)} → {len(target_keypoints)})")
    
    ref_vectors = []
    for kp in ref_keypoints:
        if kp:
            ref_vectors.append(keypoints_to_vector(kp))
        else:
            ref_vectors.append(np.zeros(38))
    
    target_vectors = []
    for kp in target_keypoints:
        if kp:
            target_vectors.append(keypoints_to_vector(kp))
        else:
            target_vectors.append(np.zeros(38))
    
    ref_vectors = np.array(ref_vectors)
    target_vectors = np.array(target_vectors)
    
    distance, path = fastdtw(ref_vectors, target_vectors, dist=euclidean)
    
    aligned_indices = []
    
    for ref_idx in range(len(ref_keypoints)):
        matching = [target_idx for r_idx, target_idx in path if r_idx == ref_idx]
        
        if matching:
            aligned_indices.append(matching[0])
        else:
            if aligned_indices:
                aligned_indices.append(aligned_indices[-1])
            else:
                aligned_indices.append(0)
    
    print(f"      ✅ DTW 완료 (거리: {distance:.2f})")
    
    return aligned_indices


def draw_skeleton_on_black(keypoints: dict, img_width: int = 400, 
                           img_height: int = 600, 
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
        
        cv2.circle(canvas, (px, py), 5, color, -1)
    
    return canvas


def create_comparison_video(all_gt_keypoints: list, gt_names: list, 
                           gt_keyframes: list, output_path: str, 
                           reference_idx: int = 0):
    """DTW 동기화 비교 동영상 생성"""
    
    print("\n[동영상 생성 중...]")
    
    ref_keypoints = all_gt_keypoints[reference_idx]
    ref_name = gt_names[reference_idx]
    
    print(f"   기준 GT: {ref_name} ({len(ref_keypoints)}프레임)")
    
    # DTW 정렬
    aligned_indices_list = []
    
    for i, (keypoints, name) in enumerate(zip(all_gt_keypoints, gt_names)):
        print(f"\n   [{i+1}/{len(gt_names)}] {name}")
        
        if i == reference_idx:
            aligned_indices = list(range(len(keypoints)))
        else:
            aligned_indices = compute_dtw_alignment(ref_keypoints, keypoints)
        
        aligned_indices_list.append(aligned_indices)
    
    # 동영상 설정
    num_gts = len(gt_names)
    frame_width = 400
    frame_height = 600
    
    # 2x2 또는 1x4 레이아웃
    if num_gts <= 2:
        cols = num_gts
        rows = 1
    elif num_gts <= 4:
        cols = 2
        rows = 2
    else:
        cols = 3
        rows = (num_gts + 2) // 3
    
    output_width = frame_width * cols + 20 * (cols - 1)
    output_height = frame_height * rows + 20 * (rows - 1) + 100
    
    fps = 30.0
    
    # 색상
    colors = [
        (0, 255, 0),      # 초록
        (255, 100, 0),    # 파랑
        (0, 100, 255),    # 빨강
        (255, 255, 0),    # 청록
    ]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (output_width, output_height))
    
    total_frames = len(ref_keypoints)
    
    print(f"\n   출력: {total_frames}프레임, {fps:.2f}fps")
    
    for ref_frame_id in range(total_frames):
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 상단 헤더
        cv2.rectangle(canvas, (0, 0), (output_width, 80), (30, 30, 30), -1)
        cv2.putText(canvas, f"Frame: {ref_frame_id} (DTW Aligned)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Time: {ref_frame_id/fps:.2f}s", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 각 GT 그리기
        for i, (keypoints_list, name, kfs, aligned_indices, color) in enumerate(
            zip(all_gt_keypoints, gt_names, gt_keyframes, aligned_indices_list, colors)):
            
            target_frame_id = aligned_indices[ref_frame_id]
            
            if target_frame_id < len(keypoints_list):
                keypoints = keypoints_list[target_frame_id]
            else:
                keypoints = {}
            
            if keypoints:
                skeleton = draw_skeleton_on_black(keypoints, frame_width, frame_height, color)
            else:
                skeleton = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # 제목
            cv2.putText(skeleton, name, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(skeleton, f"F:{target_frame_id}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Key Frame 표시
            for phase, kf_id in kfs.items():
                if target_frame_id == kf_id:
                    cv2.putText(skeleton, f"[{phase}]", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.rectangle(skeleton, (0, 0), (frame_width, frame_height),
                                (0, 255, 255), 3)
            
            # 그리드 위치 계산
            row = i // cols
            col = i % cols
            
            x_offset = col * (frame_width + 20)
            y_offset = 80 + row * (frame_height + 20)
            
            canvas[y_offset:y_offset+frame_height, 
                  x_offset:x_offset+frame_width] = skeleton
        
        # 범례
        legend_y = output_height - 30
        for i, (name, color) in enumerate(zip(gt_names, colors[:len(gt_names)])):
            x_pos = 20 + i * 250
            cv2.circle(canvas, (x_pos, legend_y), 8, color, -1)
            cv2.putText(canvas, name, (x_pos + 15, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(canvas)
        
        if ref_frame_id % 10 == 0:
            progress = (ref_frame_id / total_frames) * 100
            print(f"   진행: {progress:.1f}%", end='\r')
    
    out.release()
    
    print(f"\n   ✅ 동영상 저장: {os.path.basename(output_path)}")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("GT 동영상 통합 처리 & DTW 비교")
    print("=" * 60)
    
    # GT 입력
    print("\n💡 GT 개수:")
    num_gts = int(input("   몇 개? (2~4): ").strip())
    
    gt_configs = []
    
    for i in range(num_gts):
        print(f"\n[GT {i+1}/{num_gts}]")
        
        name = input(f"   이름 (예: pro{i+1}): ").strip()
        if not name:
            name = f"GT{i+1}"
        
        video = input(f"   동영상 경로: ").strip().strip('"').strip("'")
        
        if not os.path.exists(video):
            print(f"   ❌ 파일 없음!")
            continue
        
        kf1 = int(input(f"   KF1 프레임: ").strip())
        kf2 = int(input(f"   KF2 프레임: ").strip())
        kf3 = int(input(f"   KF3 프레임: ").strip())
        
        gt_configs.append({
            'name': name,
            'video': video,
            'keyframes': {'KF1': kf1, 'KF2': kf2, 'KF3': kf3}
        })
    
    if len(gt_configs) == 0:
        print("\n❌ GT가 없습니다!")
        exit()
    
    print(f"\n✅ {len(gt_configs)}개 GT 입력 완료")
    
    # 출력 폴더
    print("\n💡 출력 폴더:")
    output_folder = input("   경로 (Enter=현재 폴더): ").strip().strip('"').strip("'")
    
    if not output_folder:
        output_folder = "."
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 1단계: Keypoint 추출 및 CSV 저장
    print("\n" + "=" * 60)
    print("[1/2] Keypoint 추출 & 정규화 CSV 저장")
    print("=" * 60)
    
    all_gt_keypoints = []
    
    for gt in gt_configs:
        csv_path = os.path.join(output_folder, f"{gt['name']}_normalized.csv")
        keypoints = extract_and_save_keypoints(gt['video'], csv_path)
        all_gt_keypoints.append(keypoints)
    
    # 2단계: DTW 비교 동영상 생성
    print("\n" + "=" * 60)
    print("[2/2] DTW 동기화 비교 동영상 생성")
    print("=" * 60)
    
    # 기준 GT 선택
    print("\n💡 기준 GT:")
    for i, gt in enumerate(gt_configs):
        print(f"   {i+1}. {gt['name']}")
    
    ref_choice = input("\n선택 (1/2/...): ").strip()
    reference_idx = int(ref_choice) - 1 if ref_choice.isdigit() else 0
    
    print(f"\n✅ 기준: {gt_configs[reference_idx]['name']}")
    
    # 동영상 생성
    video_output = os.path.join(output_folder, "comparison_dtw.mp4")
    
    gt_names = [gt['name'] for gt in gt_configs]
    gt_keyframes = [gt['keyframes'] for gt in gt_configs]
    
    create_comparison_video(all_gt_keypoints, gt_names, gt_keyframes, 
                          video_output, reference_idx)
    
    # 완료
    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)
    
    print(f"\n📁 저장된 파일:")
    print(f"\n   CSV (정규화된 좌표):")
    for gt in gt_configs:
        print(f"   - {gt['name']}_normalized.csv")
    
    print(f"\n   동영상 (DTW 동기화):")
    print(f"   - comparison_dtw.mp4")
    
    print(f"\n📂 저장 위치: {os.path.abspath(output_folder)}")