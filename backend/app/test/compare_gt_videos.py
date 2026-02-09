"""
3개 GT 동영상 비교
- 정규화된 스켈레톤을 나란히 배치
- 동기화된 재생
- Key Frame 표시
"""

import cv2
import mediapipe as mp
import numpy as np
import os


def normalize_keypoints(keypoints: dict) -> dict:
    """체형 정규화"""
    
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
    
    # 연결선 그리기
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
    
    # 관절점 그리기
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


def process_frame(frame, pose, img_width, img_height, color):
    """단일 프레임 처리"""
    
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
        
        # 스켈레톤 그리기
        skeleton_img = draw_skeleton_on_black(
            normalized, 
            img_width=img_width, 
            img_height=img_height,
            color=color
        )
        
        return skeleton_img
    
    else:
        # keypoint 없으면 검정 화면
        return np.zeros((img_height, img_width, 3), dtype=np.uint8)


def compare_gt_videos(video_paths: list, gt_names: list, gt_keyframes: list,
                     output_path: str, layout: str = 'horizontal'):
    """
    여러 GT 동영상 비교
    
    Args:
        video_paths: 동영상 경로 리스트
        gt_names: GT 이름 리스트 ['pro1', 'GT1', 'GT2']
        gt_keyframes: Key Frame 리스트 [{'KF1':26, ...}, {...}, {...}]
        output_path: 출력 동영상 경로
        layout: 'horizontal' (가로 배치) 또는 'vertical' (세로 배치)
    """
    
    print("\n" + "=" * 60)
    print("🎬 GT 동영상 비교 생성 중...")
    print("=" * 60)
    
    # 동영상 열기
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # 모든 동영상이 열렸는지 확인
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"❌ {gt_names[i]} 동영상을 열 수 없습니다!")
            return
    
    # 동영상 정보
    fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    total_frames_list = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    
    print(f"\n📊 동영상 정보:")
    for i, name in enumerate(gt_names):
        print(f"   {name}: {total_frames_list[i]}프레임, {fps_list[i]:.2f}fps")
    
    # 공통 FPS (평균)
    output_fps = np.mean(fps_list)
    
    # 최대 프레임 수
    max_frames = max(total_frames_list)
    
    print(f"\n📹 출력 동영상:")
    print(f"   FPS: {output_fps:.2f}")
    print(f"   총 프레임: {max_frames}개")
    print(f"   배치: {layout}")
    
    # MediaPipe 설정 (각 동영상마다)
    mp_pose = mp.solutions.pose
    poses = [mp_pose.Pose(model_complexity=1) for _ in range(len(caps))]
    
    # 개별 프레임 크기
    frame_width = 480
    frame_height = 640
    
    # 출력 동영상 크기
    if layout == 'horizontal':
        # 가로 배치: [GT1 | GT2 | GT3]
        output_width = frame_width * len(caps) + 20 * (len(caps) - 1)  # 간격 20px
        output_height = frame_height + 100  # 상단 헤더 + 하단 여백
    else:
        # 세로 배치
        output_width = frame_width + 100
        output_height = frame_height * len(caps) + 20 * (len(caps) - 1) + 100
    
    # 색상 (각 GT마다 다른 색)
    colors = [
        (0, 255, 0),    # 초록 (pro1)
        (255, 100, 0),  # 파랑 (GT1)
        (0, 100, 255)   # 빨강 (GT2)
    ]
    
    # 동영상 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                         (output_width, output_height))
    
    print(f"\n💾 처리 중...")
    
    frame_id = 0
    
    while frame_id < max_frames:
        # 전체 캔버스 (검정)
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 상단 헤더
        cv2.rectangle(canvas, (0, 0), (output_width, 80), (30, 30, 30), -1)
        cv2.putText(canvas, f"Frame: {frame_id}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Time: {frame_id/output_fps:.2f}s", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 각 GT 처리
        for i, (cap, pose, name, kfs) in enumerate(zip(caps, poses, gt_names, gt_keyframes)):
            
            # 프레임 읽기
            if frame_id < total_frames_list[i]:
                ret, frame = cap.read()
            else:
                ret = False
            
            if ret:
                # 스켈레톤 생성
                skeleton = process_frame(frame, pose, frame_width, frame_height, colors[i])
                
                # 제목 추가
                cv2.putText(skeleton, name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
                
                # Key Frame 표시
                for phase, kf_id in kfs.items():
                    if frame_id == kf_id:
                        cv2.putText(skeleton, f"[{phase}]", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.rectangle(skeleton, (0, 0), (frame_width, frame_height),
                                    (0, 255, 255), 3)
            else:
                # 프레임 없으면 검정
                skeleton = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(skeleton, f"{name} (END)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
            
            # 캔버스에 배치
            if layout == 'horizontal':
                x_offset = i * (frame_width + 20)
                y_offset = 80
                canvas[y_offset:y_offset+frame_height, 
                      x_offset:x_offset+frame_width] = skeleton
            else:
                x_offset = 50
                y_offset = 80 + i * (frame_height + 20)
                canvas[y_offset:y_offset+frame_height, 
                      x_offset:x_offset+frame_width] = skeleton
        
        # 하단 범례
        legend_y = output_height - 30
        for i, (name, color) in enumerate(zip(gt_names, colors)):
            cv2.circle(canvas, (20 + i*200, legend_y), 8, color, -1)
            cv2.putText(canvas, name, (35 + i*200, legend_y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 동영상에 쓰기
        out.write(canvas)
        
        # 진행 상황
        if frame_id % 10 == 0:
            progress = (frame_id / max_frames) * 100
            print(f"   진행: {progress:.1f}% ({frame_id}/{max_frames})")
        
        frame_id += 1
    
    # 정리
    for cap in caps:
        cap.release()
    
    for pose in poses:
        pose.close()
    
    out.release()
    
    print(f"\n✅ 비교 동영상 생성 완료!")
    print(f"   저장: {output_path}")
    print(f"   크기: {output_width}x{output_height}")
    print(f"   FPS: {output_fps:.2f}")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("GT 동영상 비교 생성")
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
    
    print("\n💡 사용할 GT 선택:")
    print("   1. 3개 모두 (pro1, GT1, GT2)")
    print("   2. 수동 선택")
    
    choice = input("\n선택 (1/2): ").strip()
    
    if choice == '1':
        selected_gts = gt_configs
    else:
        selected_gts = []
        for gt in gt_configs:
            use = input(f"\n{gt['name']} 사용? (y/n): ").strip().lower()
            if use == 'y':
                selected_gts.append(gt)
    
    if len(selected_gts) == 0:
        print("\n❌ 선택된 GT가 없습니다!")
        exit()
    
    print(f"\n✅ {len(selected_gts)}개 GT 선택됨")
    
    # 동영상 경로 확인
    for gt in selected_gts:
        if not os.path.exists(gt['video']):
            print(f"❌ {gt['name']} 동영상 없음: {gt['video']}")
            exit()
    
    # 배치 방식
    print("\n💡 배치 방식:")
    print("   1. 가로 배치 (추천) [GT1 | GT2 | GT3]")
    print("   2. 세로 배치")
    
    layout_choice = input("\n선택 (1/2): ").strip()
    layout = 'vertical' if layout_choice == '2' else 'horizontal'
    
    # 출력 경로
    print("\n💡 출력 경로:")
    default_output = r"C:\Users\User\Desktop\CV\FinalProj\data\gt_comparison.mp4"
    print(f"   기본: {default_output}")
    
    output_path = input("\n출력 경로 (Enter=기본): ").strip().strip('"').strip("'")
    
    if not output_path:
        output_path = default_output
    
    print(f"\n✅ 출력: {output_path}")
    
    # 실행
    video_paths = [gt['video'] for gt in selected_gts]
    gt_names = [gt['name'] for gt in selected_gts]
    gt_keyframes = [gt['keyframes'] for gt in selected_gts]
    
    compare_gt_videos(video_paths, gt_names, gt_keyframes, output_path, layout)
    
    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)