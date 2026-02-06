"""
3개 GT 동영상 DTW 동기화 비교
- DTW로 스윙 속도 정렬
- 검정 배경 스켈레톤
- Key Frame 동기화
"""

import cv2
import mediapipe as mp
import numpy as np
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


def keypoints_to_vector(keypoints: dict) -> np.ndarray:
    """
    Keypoint를 1차원 벡터로 변환 (DTW용)
    
    Returns:
        [x1, y1, x2, y2, ...] 형태의 벡터
    """
    
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


def extract_all_keypoints(video_path: str) -> list:
    """
    동영상의 모든 프레임에서 keypoint 추출 및 정규화
    
    Returns:
        [keypoints_frame0, keypoints_frame1, ...]
    """
    
    print(f"\n   📹 {os.path.basename(video_path)} keypoint 추출 중...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"      ❌ 동영상 열기 실패!")
        return []
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    
    all_keypoints = []
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
        else:
            # keypoint 없으면 빈 딕셔너리
            all_keypoints.append({})
        
        frame_id += 1
    
    cap.release()
    pose.close()
    
    print(f"      ✅ {len(all_keypoints)}개 프레임 추출 완료")
    
    return all_keypoints


def compute_dtw_alignment(ref_keypoints: list, target_keypoints: list) -> list:
    """
    DTW로 두 시퀀스 정렬
    
    Args:
        ref_keypoints: 기준 keypoint 시퀀스
        target_keypoints: 정렬할 keypoint 시퀀스
    
    Returns:
        aligned_indices: ref의 각 프레임에 대응하는 target 프레임 인덱스
    """
    
    print(f"\n   🔄 DTW 정렬 중...")
    print(f"      기준: {len(ref_keypoints)}프레임")
    print(f"      대상: {len(target_keypoints)}프레임")
    
    # Keypoint를 벡터로 변환
    ref_vectors = []
    for kp in ref_keypoints:
        if kp:  # 빈 딕셔너리 아니면
            ref_vectors.append(keypoints_to_vector(kp))
        else:
            ref_vectors.append(np.zeros(38))  # 19개 keypoint * 2 (x, y)
    
    target_vectors = []
    for kp in target_keypoints:
        if kp:
            target_vectors.append(keypoints_to_vector(kp))
        else:
            target_vectors.append(np.zeros(38))
    
    ref_vectors = np.array(ref_vectors)
    target_vectors = np.array(target_vectors)
    
    # DTW 계산
    distance, path = fastdtw(ref_vectors, target_vectors, dist=euclidean)
    
    print(f"      DTW 거리: {distance:.2f}")
    
    # 정렬 경로에서 인덱스 매핑 추출
    aligned_indices = []
    
    for ref_idx in range(len(ref_keypoints)):
        # ref_idx에 대응하는 target_idx 찾기
        matching = [target_idx for r_idx, target_idx in path if r_idx == ref_idx]
        
        if matching:
            # 여러 개면 첫 번째 사용
            aligned_indices.append(matching[0])
        else:
            # 대응 없으면 이전 값 사용
            if aligned_indices:
                aligned_indices.append(aligned_indices[-1])
            else:
                aligned_indices.append(0)
    
    print(f"      ✅ 정렬 완료")
    print(f"      속도 비율: {len(target_keypoints)/len(ref_keypoints):.2f}x")
    
    return aligned_indices


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


def compare_gt_videos_dtw(video_paths: list, gt_names: list, gt_keyframes: list,
                          output_path: str, reference_idx: int = 0):
    """
    DTW 기반 GT 동영상 비교
    
    Args:
        video_paths: 동영상 경로 리스트
        gt_names: GT 이름 리스트
        gt_keyframes: Key Frame 리스트
        output_path: 출력 동영상 경로
        reference_idx: 기준이 되는 GT 인덱스 (0=pro1)
    """
    
    print("\n" + "=" * 60)
    print("🎬 DTW 기반 GT 동영상 비교 생성")
    print("=" * 60)
    
    # 1단계: 모든 GT의 keypoint 추출
    print("\n[1/3] 모든 GT keypoint 추출 중...")
    
    all_gt_keypoints = []
    
    for i, (path, name) in enumerate(zip(video_paths, gt_names)):
        keypoints = extract_all_keypoints(path)
        all_gt_keypoints.append(keypoints)
    
    # 2단계: DTW 정렬
    print("\n[2/3] DTW 정렬 중...")
    
    ref_keypoints = all_gt_keypoints[reference_idx]
    ref_name = gt_names[reference_idx]
    
    print(f"\n   기준 GT: {ref_name} ({len(ref_keypoints)}프레임)")
    
    # 각 GT를 기준 GT에 정렬
    aligned_indices_list = []
    
    for i, (keypoints, name) in enumerate(zip(all_gt_keypoints, gt_names)):
        if i == reference_idx:
            # 기준 GT는 그대로
            aligned_indices = list(range(len(keypoints)))
        else:
            # DTW 정렬
            aligned_indices = compute_dtw_alignment(ref_keypoints, keypoints)
        
        aligned_indices_list.append(aligned_indices)
    
    # 3단계: 동영상 생성
    print("\n[3/3] 동영상 생성 중...")
    
    # 출력 설정
    frame_width = 480
    frame_height = 640
    output_width = frame_width * len(video_paths) + 20 * (len(video_paths) - 1)
    output_height = frame_height + 100
    
    fps = 30.0
    
    colors = [
        (0, 255, 0),    # 초록
        (255, 100, 0),  # 파랑
        (0, 100, 255)   # 빨강
    ]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (output_width, output_height))
    
    total_frames = len(ref_keypoints)
    
    print(f"\n   출력 프레임 수: {total_frames}개")
    print(f"   FPS: {fps:.2f}")
    
    for ref_frame_id in range(total_frames):
        # 전체 캔버스
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
            
            # 정렬된 프레임 인덱스
            target_frame_id = aligned_indices[ref_frame_id]
            
            # Keypoint 가져오기
            if target_frame_id < len(keypoints_list):
                keypoints = keypoints_list[target_frame_id]
            else:
                keypoints = {}
            
            # 스켈레톤 그리기
            if keypoints:
                skeleton = draw_skeleton_on_black(keypoints, frame_width, frame_height, color)
            else:
                skeleton = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # 제목
            cv2.putText(skeleton, name, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(skeleton, f"F:{target_frame_id}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Key Frame 표시
            for phase, kf_id in kfs.items():
                if target_frame_id == kf_id:
                    cv2.putText(skeleton, f"[{phase}]", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(skeleton, (0, 0), (frame_width, frame_height),
                                (0, 255, 255), 3)
            
            # 캔버스에 배치
            x_offset = i * (frame_width + 20)
            y_offset = 80
            canvas[y_offset:y_offset+frame_height, 
                  x_offset:x_offset+frame_width] = skeleton
        
        # 하단 범례
        legend_y = output_height - 30
        for i, (name, color) in enumerate(zip(gt_names, colors)):
            cv2.circle(canvas, (20 + i*200, legend_y), 8, color, -1)
            cv2.putText(canvas, name, (35 + i*200, legend_y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 쓰기
        out.write(canvas)
        
        # 진행 상황
        if ref_frame_id % 10 == 0:
            progress = (ref_frame_id / total_frames) * 100
            print(f"   진행: {progress:.1f}% ({ref_frame_id}/{total_frames})")
    
    out.release()
    
    print(f"\n✅ DTW 동기화 동영상 생성 완료!")
    print(f"   저장: {output_path}")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("DTW 동기화 GT 비교 생성")
    print("=" * 60)
    
    # GT 데이터
    gt_configs = [
        {
            'name': 'pro1',
            'video': r"C:\Users\User\Desktop\CV\FinalProj\data\pro1.mp4",
            'keyframes': {'KF1': 26, 'KF2': 45, 'KF3': 57}
        },
        {
            'name': 'GT1',
            'video': r"C:\Users\User\Desktop\CV\FinalProj\data\GT1.mp4",
            'keyframes': {'KF1': 44, 'KF2': 56, 'KF3': 64}
        },
        {
            'name': 'GT2',
            'video': r"C:\Users\User\Desktop\CV\FinalProj\data\GT2.mp4",
            'keyframes': {'KF1': 39, 'KF2': 51, 'KF3': 58}
        }
    ]
    
    # 동영상 확인
    for gt in gt_configs:
        if not os.path.exists(gt['video']):
            print(f"❌ {gt['name']} 동영상 없음: {gt['video']}")
            exit()
    
    # 기준 GT 선택
    print("\n💡 기준 GT 선택 (다른 GT들이 이 GT에 맞춰 정렬됩니다):")
    for i, gt in enumerate(gt_configs):
        print(f"   {i+1}. {gt['name']}")
    
    ref_choice = input("\n선택 (1/2/3): ").strip()
    reference_idx = int(ref_choice) - 1 if ref_choice in ['1', '2', '3'] else 0
    
    print(f"\n✅ 기준 GT: {gt_configs[reference_idx]['name']}")
    
    # 출력 경로
    print("\n💡 출력 경로:")
    default_output = r"C:\Users\User\Desktop\CV\FinalProj\data\gt_comparison_dtw.mp4"
    print(f"   기본: {default_output}")
    
    output_path = input("\n출력 경로 (Enter=기본): ").strip().strip('"').strip("'")
    
    if not output_path:
        output_path = default_output
    
    # 실행
    video_paths = [gt['video'] for gt in gt_configs]
    gt_names = [gt['name'] for gt in gt_configs]
    gt_keyframes = [gt['keyframes'] for gt in gt_configs]
    
    compare_gt_videos_dtw(video_paths, gt_names, gt_keyframes, 
                         output_path, reference_idx)
    
    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)
    print("\n💡 이제 3개 GT가 같은 스윙 타이밍으로 동기화되어 있습니다!")