import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import pandas as pd
import numpy as np
from app.services.swing.engine import PoseDetector, KeyframeDetector
import math

class GTMetricsGenerator:
    """GT 영상에서 메트릭 추출"""
    
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.keyframe_detector = KeyframeDetector()
    
    def get_angle_3pt(self, p1, p2, p3):
        """3점 각도 계산"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 0
        
        cos_angle = np.dot(ba, bc) / norm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))
    
    def get_line_angle(self, p1, p2):
        """2점 직선 각도"""
        dx = float(p2[0]) - float(p1[0])
        dy = float(p1[1]) - float(p2[1])
        return abs(math.degrees(math.atan2(dy, dx)))
    
    def extract_from_video(self, video_path: str):
        """
        GT 영상에서 메트릭 추출
        
        Returns:
            {
                'Ready_Elbow_Height': 0.018,
                'Backswing_Angle': 122.5,
                'Impact_Arm_Angle': 58.5,
                'Impact_Rotation_Delta': 35.0
            }
        """
        print(f"\n{'='*50}")
        print(f"🎥 GT 영상 분석 시작: {video_path}")
        print(f"{'='*50}\n")
        
        # 1. 영상에서 keypoint 추출
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        frame_id = 0
        
        print("📹 프레임별 keypoint 추출 중...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Keypoint 추출
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                kp = {}
                kp['frame_id'] = frame_id
                
                # 19개 keypoint 추출
                selected = {
                    0: 'nose', 11: 'left_shoulder', 12: 'right_shoulder',
                    13: 'left_elbow', 14: 'right_elbow', 15: 'left_wrist',
                    16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky',
                    23: 'left_hip', 24: 'right_hip', 25: 'left_knee',
                    26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle',
                    29: 'left_heel', 30: 'right_heel', 31: 'left_foot_index',
                    32: 'right_foot_index'
                }
                
                for idx, name in selected.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    kp[f'{name}_x'] = landmark.x
                    kp[f'{name}_y'] = landmark.y
                    kp[f'{name}_z'] = landmark.z
                    kp[f'{name}_visibility'] = landmark.visibility
                
                keypoints_list.append(kp)
            
            frame_id += 1
        
        cap.release()
        pose.close()
        
        print(f"✅ 총 {len(keypoints_list)}개 프레임 추출 완료\n")
        
        if not keypoints_list:
            print("❌ Keypoint를 찾을 수 없습니다!")
            return None
        
        # 2. DataFrame 생성
        df = pd.DataFrame(keypoints_list)
        
        # 3. 키프레임 감지
        print("🔍 키프레임 감지 중...")
        keyframes = self.keyframe_detector.detect(df)
        
        if not keyframes:
            print("❌ 키프레임 감지 실패!")
            return None
        
        print(f"✅ 키프레임: {keyframes}\n")
        
        # 4. 메트릭 계산
        print("📊 메트릭 계산 중...")
        
        metrics = {}
        
        # Ready - 팔꿈치 높이
        ready_row = df.iloc[keyframes['ready']]
        metrics['Ready_Elbow_Height'] = (
            ready_row['right_shoulder_y'] - ready_row['right_elbow_y']
        )
        
        # Backswing - 백스윙 각도
        bs_row = df.iloc[keyframes['backswing']]
        metrics['Backswing_Angle'] = self.get_angle_3pt(
            [bs_row['right_shoulder_x'], bs_row['right_shoulder_y']],
            [bs_row['right_elbow_x'], bs_row['right_elbow_y']],
            [bs_row['right_wrist_x'], bs_row['right_wrist_y']]
        )
        
        # Impact - 팔 각도
        impact_row = df.iloc[keyframes['impact']]
        metrics['Impact_Arm_Angle'] = self.get_line_angle(
            [impact_row['right_shoulder_x'], impact_row['right_shoulder_y']],
            [impact_row['right_wrist_x'], impact_row['right_wrist_y']]
        )
        
        # Impact - 골반 회전량
        bs_hip_angle = self.get_line_angle(
            [bs_row['left_hip_x'], bs_row['left_hip_y']],
            [bs_row['right_hip_x'], bs_row['right_hip_y']]
        )
        
        impact_hip_angle = self.get_line_angle(
            [impact_row['left_hip_x'], impact_row['left_hip_y']],
            [impact_row['right_hip_x'], impact_row['right_hip_y']]
        )
        
        metrics['Impact_Rotation_Delta'] = abs(impact_hip_angle - bs_hip_angle)
        
        print("✅ 메트릭 계산 완료!\n")
        print("📋 결과:")
        for key, value in metrics.items():
            print(f"  - {key}: {value:.4f}")
        
        return metrics


def main():
    """메인 실행 함수"""
    
    # GT 영상 경로 (수정 필요)
    gt_video_path = input("GT 영상 경로를 입력하세요: ").strip()
    
    if not Path(gt_video_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {gt_video_path}")
        return
    
    # 메트릭 추출
    generator = GTMetricsGenerator()
    metrics = generator.extract_from_video(gt_video_path)
    
    if not metrics:
        print("\n❌ 메트릭 추출 실패!")
        return
    
    # CSV 저장
    output_path = "backend/data/standard/GT_angle/gt_total_metrics2.csv"
    
    df = pd.DataFrame([{
        'GT_Name': 'FILTERED_AVERAGE',
        **metrics
    }])
    
    # 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"✅ GT 메트릭 저장 완료!")
    print(f"📁 경로: {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()