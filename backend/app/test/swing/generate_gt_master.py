"""
GT 영상 10개 → 정규화 + 강력한 보간 + 하나의 CSV 통합
로컬 폴더에 저장
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from typing import Dict, List


# ========================================
# 1. 정규화
# ========================================

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
    
    return normalized


# ========================================
# 2. 강력한 보간 (무조건 적용!)
# ========================================

def interpolate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 keypoint 컬럼에 강력한 보간 적용
    - 0값 → NaN → 선형/3차 스플라인 보간
    - 매우 작은 값(< 0.0001) → NaN → 보간
    - NaN → 보간
    - 평균값으로 최종 대체
    """
    
    keypoint_cols = [col for col in df.columns
                     if (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))]
    
    print(f"   🔧 보간 적용 중... (대상 컬럼: {len(keypoint_cols)}개)")
    
    zero_before = 0
    zero_after = 0
    
    for col in keypoint_cols:
        zero_before += (df[col] == 0).sum()
        
        # 1단계: 0을 NaN으로
        df[col] = df[col].replace(0, np.nan)
        
        # 2단계: 매우 작은 값도 NaN으로
        df.loc[(df[col] > 0) & (df[col] < 0.0001), col] = np.nan
        
        # 3단계: 선형 보간 (앞뒤 모두)
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # 4단계: 3차 스플라인 보간 (부드럽게)
        if df[col].isna().sum() > 0:
            try:
                df[col] = df[col].interpolate(method='cubic', limit_direction='both')
            except:
                pass  # cubic 실패 시 넘어감
        
        # 5단계: 앞/뒤 채우기
        df[col] = df[col].ffill().bfill()
        
        # 6단계: 여전히 NaN이면 컬럼 평균값
        if df[col].isna().sum() > 0:
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.5  # 기본값
            df[col] = df[col].fillna(mean_val)
        
        # 7단계: 최종 0 체크 및 제거
        if (df[col] == 0).sum() > 0:
            non_zero_mean = df[df[col] != 0][col].mean()
            if pd.isna(non_zero_mean):
                non_zero_mean = 0.5
            df.loc[df[col] == 0, col] = non_zero_mean
        
        zero_after += (df[col] == 0).sum()
    
    print(f"   ✅ 보간 완료:")
    print(f"      0 값: {zero_before:,}개 → {zero_after:,}개")
    
    if zero_after > 0:
        print(f"      ⚠️ 경고: {zero_after}개의 0값이 남아있습니다!")
    
    return df


# ========================================
# 3. 단일 영상 처리
# ========================================

def process_single_video(video_path: str, gt_id: str) -> pd.DataFrame:
    """
    단일 GT 영상 → keypoint 추출 → 정규화 → DataFrame
    """
    
    print(f"\n   🎬 처리 중: {os.path.basename(video_path)} (GT ID: {gt_id})")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"   ❌ 동영상 열기 실패")
        return None
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    
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
            
            # 정규화
            normalized = normalize_keypoints(keypoints)
            
            # DataFrame용 데이터
            row = {
                'gt_id': gt_id,
                'frame_id': frame_id, 
                'timestamp': frame_id / fps
            }
            row.update(normalized)
            all_data.append(row)
        
        frame_id += 1
        
        # 진행 상황 (10프레임마다)
        if frame_id % 10 == 0:
            print(f"      → {frame_id}프레임...", end='\r')
    
    cap.release()
    pose.close()
    
    df = pd.DataFrame(all_data)
    
    print(f"   ✅ 완료: {frame_id}프레임                    ")
    
    return df


# ========================================
# 4. 메인: 10개 영상 통합
# ========================================

def process_multiple_gts(video_paths: List[str], output_path: str):
    """
    GT 영상 여러 개 → 하나의 CSV로 통합
    """
    
    print("\n" + "=" * 60)
    print("🎯 GT 영상 통합 처리 (보간 무조건 적용)")
    print("=" * 60)
    print(f"입력 영상 개수: {len(video_paths)}개")
    
    all_dfs = []
    
    for i, video_path in enumerate(video_paths, 1):
        gt_id = f"GT{i}"
        
        df = process_single_video(video_path, gt_id)
        
        if df is not None and len(df) > 0:
            df = interpolate_all(df)
            all_dfs.append(df)
    
    if not all_dfs:
        print("\n❌ 처리된 영상이 없습니다!")
        return
    
    # 통합
    print(f"\n📊 통합 중...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['gt_id', 'frame_id']).reset_index(drop=True)
    
    # ⭐ 데이터 정리
    print(f"\n🔧 데이터 정리 중...")
    
    # 1. x, y, z 컬럼만 남기기 (visibility 제거는 자동)
    keep_cols = ['gt_id', 'frame_id', 'timestamp']
    
    keypoint_cols = [col for col in combined_df.columns 
                     if col.endswith('_x') or col.endswith('_y') or col.endswith('_z')]
    
    keep_cols.extend(keypoint_cols)
    
    combined_df = combined_df[keep_cols]
    
    print(f"   ✅ x, y, z 컬럼만 유지 ({len(keypoint_cols)}개)")
    
    # 2. 소수점 4자리로 반올림
    for col in keypoint_cols:
        combined_df[col] = combined_df[col].round(4)
    
    print(f"   ✅ 소수점 4자리로 정리")
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"\n📄 CSV 파일: {output_path}")
    print(f"총 행 수: {len(combined_df):,}개")
    print(f"총 컬럼: {len(combined_df.columns)}개")
    
    print("\nGT별 프레임 수:")
    for gt_id in combined_df['gt_id'].unique():
        count = len(combined_df[combined_df['gt_id'] == gt_id])
        print(f"  {gt_id}: {count}프레임")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("GT 영상 통합 처리 (최대 10개)")
    print("=" * 60)
    
    video_paths = []
    
    print("\n💡 GT 영상 경로를 입력하세요 (최대 10개, 빈 입력 시 종료):\n")
    
    for i in range(1, 11):
        path = input(f"GT{i} 영상: ").strip().strip('"').strip("'")
        
        if not path:
            break
        
        if not os.path.exists(path):
            print(f"   ⚠️ 파일 없음: {path} (건너뜀)")
            continue
        
        video_paths.append(path)
    
    if not video_paths:
        print("\n❌ 입력된 영상이 없습니다!")
        exit()
    
    print(f"\n✅ 총 {len(video_paths)}개 영상 입력됨")
    
    # ⭐ 출력 경로 (로컬 폴더)
    print("\n💡 출력 폴더를 입력하세요:")
    
    # 첫 번째 영상과 같은 폴더를 기본값으로
    default_folder = os.path.dirname(video_paths[0])
    print(f"   기본: {default_folder}")
    
    output_folder = input("출력 폴더 (Enter=기본): ").strip().strip('"').strip("'")
    
    if not output_folder:
        output_folder = default_folder
    
    # 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 출력 파일명
    print("\n💡 출력 파일명을 입력하세요:")
    print(f"   기본: gt_master.csv")
    
    filename = input("파일명 (Enter=기본): ").strip()
    
    if not filename:
        filename = "gt_master.csv"
    elif not filename.endswith('.csv'):
        filename = filename + ".csv"
    
    # 최종 경로
    output_path = os.path.join(output_folder, filename)
    
    print(f"\n📍 저장 경로: {output_path}")
    
    # 실행
    process_multiple_gts(video_paths, output_path)