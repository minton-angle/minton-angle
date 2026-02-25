"""
전처리: 정규화 + 강력한 보간
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import Dict, List


class Preprocessor:
    """영상 전처리 (정규화 + 보간)"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1)
        
        self.keypoint_indices = {
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
    
    def normalize_keypoints(self, keypoints: dict) -> dict:
        """
        체형 정규화 (어깨 너비 + 골반 중심)
        
        Args:
            keypoints: {
                'left_shoulder_x': 0.5, 'left_shoulder_y': 0.3, ...
            }
            
        Returns:
            정규화된 keypoints
        """
        
        left_shoulder = np.array([
            keypoints['left_shoulder_x'], 
            keypoints['left_shoulder_y']
        ])
        right_shoulder = np.array([
            keypoints['right_shoulder_x'], 
            keypoints['right_shoulder_y']
        ])
        
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        if shoulder_width < 0.01:
            shoulder_width = 0.1
        
        left_hip = np.array([
            keypoints['left_hip_x'], 
            keypoints['left_hip_y']
        ])
        right_hip = np.array([
            keypoints['right_hip_x'], 
            keypoints['right_hip_y']
        ])
        
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
    
    def interpolate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        강력한 보간 (0값 제거)
        
        Args:
            df: keypoints DataFrame
            
        Returns:
            보간된 DataFrame
        """
        
        keypoint_cols = [
            col for col in df.columns
            if (col.endswith('_x') or col.endswith('_y') or col.endswith('_z'))
            and not col.endswith('_visibility')
        ]
        
        print(f"   🔧 보간 적용 중... (대상 컬럼: {len(keypoint_cols)}개)")
        
        zero_before = 0
        zero_after = 0
        
        for col in keypoint_cols:
            zero_before += (df[col] == 0).sum()
            
            # 1단계: 0을 NaN으로
            df[col] = df[col].replace(0, np.nan)
            
            # 2단계: 매우 작은 값도 NaN으로
            df.loc[(df[col] > 0) & (df[col] < 0.0001), col] = np.nan
            
            # 3단계: 선형 보간
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # 4단계: 3차 스플라인 보간
            if df[col].isna().sum() > 0:
                try:
                    df[col] = df[col].interpolate(method='cubic', limit_direction='both')
                except:
                    pass
            
            # 5단계: 앞/뒤 채우기
            df[col] = df[col].ffill().bfill()
            
            # 6단계: 평균값으로 최종 대체
            if df[col].isna().sum() > 0:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.5
                df[col] = df[col].fillna(mean_val)
            
            # 7단계: 최종 0 체크 및 제거
            if (df[col] == 0).sum() > 0:
                non_zero_mean = df[df[col] != 0][col].mean()
                if pd.isna(non_zero_mean):
                    non_zero_mean = 0.5
                df.loc[df[col] == 0, col] = non_zero_mean
            
            zero_after += (df[col] == 0).sum()
        
        print(f"   ✅ 보간 완료: 0값 {zero_before:,}개 → {zero_after:,}개")
        
        return df
    
    def process_video(self, video_path: str) -> pd.DataFrame:
        """
        영상 처리: keypoint 추출 → 정규화 → DataFrame
        
        Args:
            video_path: 영상 파일 경로
            
        Returns:
            DataFrame with columns: frame_id, timestamp, nose_x, nose_y, ...
        """
        
        print(f"\n   🎬 영상 처리 중: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"   ❌ 동영상 열기 실패")
            return None
        
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
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                keypoints = {}
                
                for idx, name in self.keypoint_indices.items():
                    lm = landmarks[idx]
                    keypoints[f'{name}_x'] = lm.x
                    keypoints[f'{name}_y'] = lm.y
                    keypoints[f'{name}_z'] = lm.z
                    keypoints[f'{name}_visibility'] = lm.visibility
                
                # 정규화
                normalized = self.normalize_keypoints(keypoints)
                
                # DataFrame용 데이터
                row = {
                    'frame_id': frame_id, 
                    'timestamp': frame_id / fps
                }
                row.update(normalized)
                all_data.append(row)
            
            frame_id += 1
            
            if frame_id % 10 == 0:
                print(f"      → {frame_id}프레임...", end='\r')
        
        cap.release()
        
        df = pd.DataFrame(all_data)
        
        print(f"   ✅ 완료: {frame_id}프레임                    ")
        
        # 보간 적용
        df = self.interpolate_all(df)
        
        return df
    
    def __del__(self):
        """소멸자: MediaPipe 리소스 해제"""
        if hasattr(self, 'pose'):
            self.pose.close()

    # ==============================================================================
# [실행부] 이 부분을 코드 맨 아래에 추가하세요!
# ==============================================================================
if __name__ == "__main__":
    # 1. 전처리 기능(클래스) 꺼내오기
    preprocessor = Preprocessor()
    
    # 2. 분석할 영상 경로 지정 (여기에 경로를 넣으시면 됩니다!)
    VIDEO_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/junseo/junseo.mp4'
    
    # 3. 영상 처리 실행 (DataFrame 형태로 결과가 나옵니다)
    result_df = preprocessor.process_video(VIDEO_PATH)
    
    # 4. 결과 확인 및 CSV 저장
    if result_df is not None:
        print("\n📊 전처리된 데이터 미리보기:")
        print(result_df.head()) # 데이터 앞부분 살짝 확인
        
        # CSV 파일로 저장 (원하는 저장 경로로 수정하세요)
        SAVE_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/junseo/junseo.csv'
        result_df.to_csv(SAVE_PATH, index=False)
        print(f"\n🎉 전처리 성공! 데이터가 [{SAVE_PATH}]에 저장되었습니다.")