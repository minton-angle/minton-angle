"""
GT 동영상 정규화 및 시각화
- 어깨 너비 + 골반 중심 기준 정규화
- 0~1 스케일링
- 검정 배경 + 스켈레톤 시각화
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json


def normalize_keypoints(keypoints: dict) -> dict:
    """
    체형 정규화 (어깨 너비 기준, 골반 중심)
    
    Steps:
    1. 골반 중심을 원점(0, 0)으로 이동
    2. 어깨 너비로 스케일 조정
    3. 0~1 범위로 재스케일링
    
    Returns:
        정규화된 keypoint (0~1 범위)
    """
    
    # 1. 어깨 너비 계산
    left_shoulder = np.array([keypoints['left_shoulder_x'], 
                             keypoints['left_shoulder_y']])
    right_shoulder = np.array([keypoints['right_shoulder_x'], 
                              keypoints['right_shoulder_y']])
    
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    
    if shoulder_width < 0.01:
        shoulder_width = 0.1
    
    # 2. 골반 중심 계산
    left_hip = np.array([keypoints['left_hip_x'], 
                        keypoints['left_hip_y']])
    right_hip = np.array([keypoints['right_hip_x'], 
                         keypoints['right_hip_y']])
    
    hip_center = (left_hip + right_hip) / 2
    
    # 3. 모든 keypoint 정규화
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
    
    # 모든 좌표를 골반 중심 기준으로 변환
    all_x = []
    all_y = []
    
    temp_normalized = {}
    
    for name in keypoint_names:
        x = keypoints[f'{name}_x']
        y = keypoints[f'{name}_y']
        z = keypoints[f'{name}_z']
        
        # 골반 중심 기준으로 이동
        x_centered = x - hip_center[0]
        y_centered = y - hip_center[1]
        
        # 어깨 너비로 스케일 정규화
        x_norm = x_centered / shoulder_width
        y_norm = y_centered / shoulder_width
        z_norm = z / shoulder_width
        
        temp_normalized[f'{name}_x'] = x_norm
        temp_normalized[f'{name}_y'] = y_norm
        temp_normalized[f'{name}_z'] = z_norm
        
        all_x.append(x_norm)
        all_y.append(y_norm)
        
        # visibility는 그대로
        if f'{name}_visibility' in keypoints:
            temp_normalized[f'{name}_visibility'] = keypoints[f'{name}_visibility']
    
    # 4. 0~1 범위로 재스케일링
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # 정사각형 유지 (긴 축 기준)
    range_max = max(range_x, range_y)
    
    if range_max < 0.01:
        range_max = 1.0
    
    for name in keypoint_names:
        x_norm = temp_normalized[f'{name}_x']
        y_norm = temp_normalized[f'{name}_y']
        z_norm = temp_normalized[f'{name}_z']
        
        # 0~1 범위로 변환
        normalized[f'{name}_x'] = float((x_norm - min_x) / range_max)
        normalized[f'{name}_y'] = float((y_norm - min_y) / range_max)
        normalized[f'{name}_z'] = float(z_norm)  # z는 상대값 유지
        
        if f'{name}_visibility' in temp_normalized:
            normalized[f'{name}_visibility'] = temp_normalized[f'{name}_visibility']
    
    # 정규화 메타 정보
    normalized['_normalization'] = {
        'shoulder_width': float(shoulder_width),
        'hip_center_x': float(hip_center[0]),
        'hip_center_y': float(hip_center[1]),
        'scale_range': float(range_max),
        'min_x': float(min_x),
        'min_y': float(min_y),
        'method': 'shoulder_width_hip_centered_01scaled'
    }
    
    return normalized


def draw_skeleton_on_black(keypoints: dict, img_width: int = 640, 
                           img_height: int = 640, 
                           color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    검정 배경에 정규화된 스켈레톤 그리기
    
    Args:
        keypoints: 정규화된 keypoint (0~1 범위)
        img_width, img_height: 출력 이미지 크기
        color: 스켈레톤 색상 (B, G, R)
    
    Returns:
        검정 배경 이미지
    """
    
    # 검정 캔버스
    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # MediaPipe 연결선
    connections = [
        # 얼굴
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
        
        # 상체
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('right_wrist', 'right_pinky'),
        
        # 몸통
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        
        # 왼쪽 다리
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'),
        ('left_ankle', 'left_foot_index'),
        
        # 오른쪽 다리
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot_index'),
    ]
    
    # 1. 연결선 그리기
    for start_name, end_name in connections:
        start_x = keypoints.get(f'{start_name}_x')
        start_y = keypoints.get(f'{start_name}_y')
        end_x = keypoints.get(f'{end_name}_x')
        end_y = keypoints.get(f'{end_name}_y')
        
        if start_x is None or end_x is None:
            continue
        
        # 0~1 → 픽셀 좌표
        x1 = int(start_x * img_width)
        y1 = int(start_y * img_height)
        x2 = int(end_x * img_width)
        y2 = int(end_y * img_height)
        
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
    
    # 2. 관절점 그리기
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


def process_video(video_path: str, output_folder: str, 
                 keyframes: dict = None, save_all: bool = False):
    """
    동영상 처리: Keypoint 추출 → 정규화 → 시각화
    
    Args:
        video_path: 입력 동영상
        output_folder: 출력 폴더
        keyframes: {'KF1': 26, 'KF2': 45, 'KF3': 57} (None이면 모든 프레임)
        save_all: True면 모든 프레임, False면 keyframes만
    """
    
    print("\n" + "=" * 60)
    print(f"📹 처리 중: {os.path.basename(video_path)}")
    print("=" * 60)
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다!")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📊 동영상 정보:")
    print(f"   총 프레임: {total_frames}개")
    print(f"   FPS: {fps:.2f}")
    
    # MediaPipe 설정
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    
    # 데이터 저장
    all_normalized_data = []
    saved_images = {}
    
    print(f"\n💾 처리 중...")
    
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Pose estimation
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 원본 keypoint 추출
            landmarks = results.pose_landmarks.landmark
            
            keypoints = {
                'frame_id': frame_id,
                'timestamp': frame_id / fps
            }
            
            # 19개 keypoint
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
            normalized['frame_id'] = frame_id
            normalized['timestamp'] = frame_id / fps
            
            all_normalized_data.append(normalized)
            
            # 이미지 저장 조건
            should_save = save_all or (keyframes and frame_id in keyframes.values())
            
            if should_save:
                # 검정 배경에 스켈레톤 그리기
                skeleton_img = draw_skeleton_on_black(
                    normalized, 
                    img_width=640, 
                    img_height=640,
                    color=(0, 255, 0)
                )
                
                # 정보 오버레이
                cv2.putText(skeleton_img, f"Frame: {frame_id}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(skeleton_img, f"Time: {frame_id/fps:.2f}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Key Frame 표시
                if keyframes:
                    for phase, kf_id in keyframes.items():
                        if frame_id == kf_id:
                            cv2.putText(skeleton_img, f"[{phase}]", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
                
                # 저장
                img_path = os.path.join(output_folder, f"frame_{frame_id:04d}_normalized.jpg")
                cv2.imwrite(img_path, skeleton_img)
                
                saved_images[frame_id] = img_path
        
        frame_id += 1
        
        # 진행 상황 (10프레임마다)
        if frame_id % 10 == 0:
            progress = (frame_id / total_frames) * 100
            print(f"   진행: {progress:.1f}% ({frame_id}/{total_frames})")
    
    cap.release()
    pose.close()
    
    # CSV 저장
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_folder, f"{video_name}_normalized.csv")
    
    df = pd.DataFrame(all_normalized_data)
    
    # _normalization 컬럼 제거 (메타데이터)
    meta_cols = [col for col in df.columns if col.startswith('_')]
    df_save = df.drop(columns=meta_cols, errors='ignore')
    
    df_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 정규화 메타데이터 저장
    if len(all_normalized_data) > 0 and '_normalization' in all_normalized_data[0]:
        meta_path = os.path.join(output_folder, f"{video_name}_normalization_meta.json")
        
        # 첫 프레임의 정규화 정보 (모든 프레임 동일)
        meta_info = all_normalized_data[0]['_normalization']
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 처리 완료!")
    print(f"   CSV: {csv_path}")
    print(f"   프레임 수: {len(df)}개")
    print(f"   저장된 이미지: {len(saved_images)}개")
    
    if saved_images:
        print(f"\n📸 저장된 이미지:")
        for frame_id, img_path in sorted(saved_images.items())[:5]:
            print(f"      Frame {frame_id}: {os.path.basename(img_path)}")
        if len(saved_images) > 5:
            print(f"      ... 외 {len(saved_images)-5}개")
    
    return csv_path, saved_images


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("배드민턴 정규화 & 시각화")
    print("=" * 60)
    
    # GT 데이터 (미리 정의)
    gt_videos = [
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
    
    print("\n💡 처리 모드:")
    print("   1. 3개 GT 동영상 처리 (Key Frame만)")
    print("   2. 3개 GT 동영상 처리 (모든 프레임)")
    print("   3. 단일 동영상 처리")
    
    mode = input("\n선택 (1/2/3): ").strip()
    
    if mode in ['1', '2']:
        # 3개 GT 처리
        save_all = (mode == '2')
        
        print(f"\n✅ 3개 GT 처리 모드")
        print(f"   저장: {'모든 프레임' if save_all else 'Key Frame만'}")
        
        # 출력 기본 폴더
        base_output = r"C:\Users\User\Desktop\CV\FinalProj\data\normalized"
        
        for i, gt in enumerate(gt_videos, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/3] {gt['name']}")
            print(f"{'='*60}")
            
            if not os.path.exists(gt['video']):
                print(f"⚠️  동영상 파일 없음: {gt['video']}")
                continue
            
            output_folder = os.path.join(base_output, gt['name'])
            
            process_video(
                video_path=gt['video'],
                output_folder=output_folder,
                keyframes=gt['keyframes'],
                save_all=save_all
            )
        
        print("\n" + "=" * 60)
        print("🎉 모든 GT 처리 완료!")
        print("=" * 60)
        print(f"\n📁 저장 위치: {base_output}")
    
    else:
        # 단일 동영상
        print("\n💡 동영상 경로:")
        video_path = input("동영상: ").strip().strip('"').strip("'")
        
        if not os.path.exists(video_path):
            print(f"\n❌ 파일 없음!")
            exit()
        
        print("\n💡 출력 폴더:")
        output_folder = input("출력 폴더: ").strip().strip('"').strip("'")
        
        if not output_folder:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_folder = os.path.join(os.path.dirname(video_path), f"{video_name}_normalized")
        
        print("\n💡 저장 옵션:")
        print("   1. Key Frame만 (프레임 번호 입력)")
        print("   2. 모든 프레임")
        
        save_choice = input("\n선택 (1/2): ").strip()
        
        keyframes = None
        save_all = (save_choice == '2')
        
        if save_choice == '1':
            print("\n💡 Key Frame 번호 입력:")
            kf1 = int(input("   KF1: ").strip())
            kf2 = int(input("   KF2: ").strip())
            kf3 = int(input("   KF3: ").strip())
            
            keyframes = {'KF1': kf1, 'KF2': kf2, 'KF3': kf3}
        
        process_video(video_path, output_folder, keyframes, save_all)