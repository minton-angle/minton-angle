"""
점수 계산: GT 기반 평가
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Optional


class ScoreCalculator:
    """GT 기반 점수 계산"""
    
    def __init__(self, gt_metrics_path: str):
        """
        Args:
            gt_metrics_path: GT 평균 메트릭 CSV 파일 경로
                예: "backend/data/standard/GT_angle/gt_total_metrics2.csv"
        """
        # GT 평균값 로드
        gt_df = pd.read_csv(gt_metrics_path)
        self.gt_avg = gt_df[gt_df['GT_Name'] == 'FILTERED_AVERAGE'].iloc[0].to_dict()
        
        print(f"✅ GT 기준값 로드 완료: {gt_metrics_path}")
    
    def get_angle_3pt(self, p1, p2, p3) -> float:
        """
        3점 각도 계산 (p2가 꼭지점)
        
        Args:
            p1, p2, p3: [x, y] 좌표
            
        Returns:
            각도 (degree)
        """
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
    
    def get_line_angle(self, p1, p2) -> float:
        """
        2점 직선 각도 계산
        
        Args:
            p1, p2: [x, y] 좌표
            
        Returns:
            각도 (degree, 0~180)
        """
        dx = float(p2[0]) - float(p1[0])
        dy = float(p1[1]) - float(p2[1])  # y축 반전
        
        return abs(math.degrees(math.atan2(dy, dx)))
    
    def calculate_scores(
        self, 
        keypoints_df: pd.DataFrame, 
        keyframes: Dict
    ) -> Dict:
        """
        점수 계산
        
        Args:
            keypoints_df: 사용자 keypoints DataFrame
            keyframes: {'ready': 30, 'backswing': 48, 'impact': 60}
            
        Returns:
            {
                "user_evaluation": [
                    {"단계": "ready", "점수": 85.5, "부여 사유": "..."},
                    {"단계": "backswing", "점수": 72.3, "부여 사유": "..."},
                    {"단계": "impact", "점수": 90.1, "부여 사유": "..."}
                ],
                "overall_average": 82.6,
                "standard_used": "FILTERED_AVERAGE"
            }
        """
        results = []
        
        # --- [1] READY 점수 ---
        ready_idx = int(keyframes['ready'])
        ready_row = keypoints_df.iloc[ready_idx]
        
        ready_elbow_height = (
            ready_row['right_shoulder_y'] - ready_row['right_elbow_y']
        )
        
        err_h = abs(ready_elbow_height - self.gt_avg['Ready_Elbow_Height'])
        ready_score = max(0, 100 - err_h * 10000)
        
        results.append({
            '단계': 'ready',
            '점수': round(float(ready_score), 1),
            '부여 사유': f"팔꿈치 높이 오차 {err_h:.4f}"
        })
        
        # --- [2] BACKSWING 점수 ---
        backswing_idx = int(keyframes['backswing'])
        bs_row = keypoints_df.iloc[backswing_idx]
        
        bs_angle = self.get_angle_3pt(
            [bs_row['right_shoulder_x'], bs_row['right_shoulder_y']],
            [bs_row['right_elbow_x'], bs_row['right_elbow_y']],
            [bs_row['right_wrist_x'], bs_row['right_wrist_y']]
        )
        
        err_bs = abs(bs_angle - self.gt_avg['Backswing_Angle'])
        backswing_score = max(0, 100 - err_bs * 200)
        
        results.append({
            '단계': 'backswing',
            '점수': round(float(backswing_score), 1),
            '부여 사유': f"백스윙 각도 오차 {err_bs:.2f}도"
        })
        
        # --- [3] IMPACT 점수 ---
        impact_idx = int(keyframes['impact'])
        impact_row = keypoints_df.iloc[impact_idx]
        
        # 골반 회전량 계산
        bs_hip_angle = self.get_line_angle(
            [bs_row['left_hip_x'], bs_row['left_hip_y']],
            [bs_row['right_hip_x'], bs_row['right_hip_y']]
        )
        
        impact_hip_angle = self.get_line_angle(
            [impact_row['left_hip_x'], impact_row['left_hip_y']],
            [impact_row['right_hip_x'], impact_row['right_hip_y']]
        )
        
        rotation_delta = abs(impact_hip_angle - bs_hip_angle)
        rotation_threshold = self.gt_avg['Impact_Rotation_Delta'] * 0.3
        
        if rotation_delta < rotation_threshold:
            impact_score = 0.0
            impact_reason = f"골반 회전 부족 ({rotation_delta:.2f}도 < {rotation_threshold:.2f}도)"
        else:
            # 팔 각도 계산
            impact_arm_angle = self.get_line_angle(
                [impact_row['right_shoulder_x'], impact_row['right_shoulder_y']],
                [impact_row['right_wrist_x'], impact_row['right_wrist_y']]
            )
            
            err_impact = abs(impact_arm_angle - self.gt_avg['Impact_Arm_Angle'])
            impact_score = max(0, 100 - err_impact * 200)
            impact_reason = f"임팩트 팔 각도 오차 {err_impact:.2f}도"
        
        results.append({
            '단계': 'impact',
            '점수': round(float(impact_score), 1),
            '부여 사유': impact_reason
        })
        
        # --- 평균 점수 ---
        overall_avg = round(sum(item['점수'] for item in results) / len(results), 1)
        
        return {
            "user_evaluation": results,
            "overall_average": overall_avg,
            "standard_used": "FILTERED_AVERAGE"
        }
    
    def calculate_detailed_metrics(
        self,
        keypoints_df: pd.DataFrame,
        keyframes: Dict
    ) -> Dict:
        """
        상세 메트릭 계산 (LLM 피드백용)
        
        Args:
            keypoints_df: 사용자 keypoints DataFrame
            keyframes: {'ready': 30, 'backswing': 48, 'impact': 60}
            
        Returns:
            {
                'ready': {
                    'elbow_height': 0.15,
                    'gt_elbow_height': 0.18,
                    'error': 0.03
                },
                'backswing': {...},
                'impact': {...}
            }
        """
        metrics = {}
        
        # Ready
        ready_idx = int(keyframes['ready'])
        ready_row = keypoints_df.iloc[ready_idx]
        
        ready_elbow_height = (
            ready_row['right_shoulder_y'] - ready_row['right_elbow_y']
        )
        
        metrics['ready'] = {
            'elbow_height': float(ready_elbow_height),
            'gt_elbow_height': float(self.gt_avg['Ready_Elbow_Height']),
            'error': float(abs(ready_elbow_height - self.gt_avg['Ready_Elbow_Height']))
        }
        
        # Backswing
        backswing_idx = int(keyframes['backswing'])
        bs_row = keypoints_df.iloc[backswing_idx]
        
        bs_angle = self.get_angle_3pt(
            [bs_row['right_shoulder_x'], bs_row['right_shoulder_y']],
            [bs_row['right_elbow_x'], bs_row['right_elbow_y']],
            [bs_row['right_wrist_x'], bs_row['right_wrist_y']]
        )
        
        metrics['backswing'] = {
            'angle': float(bs_angle),
            'gt_angle': float(self.gt_avg['Backswing_Angle']),
            'error': float(abs(bs_angle - self.gt_avg['Backswing_Angle']))
        }
        
        # Impact
        impact_idx = int(keyframes['impact'])
        impact_row = keypoints_df.iloc[impact_idx]
        
        impact_arm_angle = self.get_line_angle(
            [impact_row['right_shoulder_x'], impact_row['right_shoulder_y']],
            [impact_row['right_wrist_x'], impact_row['right_wrist_y']]
        )
        
        bs_hip_angle = self.get_line_angle(
            [bs_row['left_hip_x'], bs_row['left_hip_y']],
            [bs_row['right_hip_x'], bs_row['right_hip_y']]
        )
        
        impact_hip_angle = self.get_line_angle(
            [impact_row['left_hip_x'], impact_row['left_hip_y']],
            [impact_row['right_hip_x'], impact_row['right_hip_y']]
        )
        
        rotation_delta = abs(impact_hip_angle - bs_hip_angle)
        
        metrics['impact'] = {
            'arm_angle': float(impact_arm_angle),
            'gt_arm_angle': float(self.gt_avg['Impact_Arm_Angle']),
            'rotation_delta': float(rotation_delta),
            'gt_rotation_delta': float(self.gt_avg['Impact_Rotation_Delta']),
            'error': float(abs(impact_arm_angle - self.gt_avg['Impact_Arm_Angle']))
        }
        
        return metrics
