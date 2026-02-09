import pandas as pd
import numpy as np
import math
import os

class BadmintonScoreCalculator:
    def __init__(self, gt_total_metrics_path):
        # 1. GT 평균 데이터 로드
        gt_df = pd.read_csv(gt_total_metrics_path)
        # GT_Name이 AVERAGE인 행만 추출하여 딕셔너리로 저장
        self.gt_avg = gt_df[gt_df['GT_Name'] == 'FILTERED_AVERAGE'].iloc[0].to_dict()
        
        # 2. MediaPipe 컬럼명 생성
        mp_parts = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
        self.columns = []
        for p in mp_parts: self.columns.extend([f"{p}_x", f"{p}_y"])

    def get_angle_3pt(self, p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0: return 0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))

    def get_line_angle(self, p1, p2):
        dx = float(p2[0]) - float(p1[0])
        dy = float(p1[1]) - float(p2[1])
        return abs(math.degrees(math.atan2(dy, dx)))

    def calculate_final_scores(self, user_points_csv, user_kf_csv, output_path):
        # 데이터 로드
        u_df = pd.read_csv(user_points_csv, header=None, names=self.columns).apply(pd.to_numeric, errors='coerce').dropna()
        u_kf = pd.read_csv(user_kf_csv).iloc[0].to_dict()
        
        results = []

        # --- [1] READY 단계 점수 ---
        u_r_idx = int(float(u_kf['ready']))
        u_ready_h = u_df.iloc[u_r_idx]['right_shoulder_y'] - u_df.iloc[u_r_idx]['right_elbow_y']
        # GT 평균 높이와의 차이 (0.1 이상 차이나면 큰 감점)
        s1 = max(0, 100 - abs(u_ready_h - self.gt_avg['Ready_Elbow_Height']) * 400)
        results.append({
            '키프레임 이름': 'ready',
            '점수': round(s1, 1),
            '측정지표': f"팔꿈치 높이차 (User: {u_ready_h:.3f} / GT_Avg: {self.gt_avg['Ready_Elbow_Height']:.3f})"
        })

        # --- [2] BACKSWING 단계 점수 ---
        u_b_idx = int(float(u_kf['backswing']))
        u_b_row = u_df.iloc[u_b_idx]
        u_bs_ang = self.get_angle_3pt(
            [u_b_row['right_shoulder_x'], u_b_row['right_shoulder_y']],
            [u_b_row['right_elbow_x'], u_b_row['right_elbow_y']],
            [u_b_row['right_wrist_x'], u_b_row['right_wrist_y']]
        )
        # GT 평균 각도와의 차이 (1도당 2점 감점)
        s2 = max(0, 100 - abs(u_bs_ang - self.gt_avg['Backswing_Angle']) * 2)
        results.append({
            '키프레임 이름': 'backswing',
            '점수': round(s2, 1),
            '측정지표': f"팔 사이각 (User: {u_bs_ang:.1f}° / GT_Avg: {self.gt_avg['Backswing_Angle']:.1f}°)"
        })

        # --- [3] IMPACT 단계 점수 (골반 회전 기반) ---
        u_i_idx = int(float(u_kf['impact']))
        u_i_row = u_df.iloc[u_i_idx]
        
        # 골반 회전량 계산 (Backswing -> Impact)
        u_bs_hip = self.get_line_angle([u_b_row['left_hip_x'], u_b_row['left_hip_y']], [u_b_row['right_hip_x'], u_b_row['right_hip_y']])
        u_i_hip = self.get_line_angle([u_i_row['left_hip_x'], u_i_row['left_hip_y']], [u_i_row['right_hip_x'], u_i_row['right_hip_y']])
        u_rot_delta = abs(u_i_hip - u_bs_hip)
        
        # [조건부 채점] 골반 회전이 GT 평균의 30% 미만이면 0점 처리
        rotation_threshold = self.gt_avg['Impact_Rotation_Delta'] * 0.3
        
        if u_rot_delta < rotation_threshold:
            s3 = 0.0
            metric_msg = f"골반 회전 부족 (회전량: {u_rot_delta:.1f}° / 최소기준: {rotation_threshold:.1f}°)"
        else:
            # 골반이 돌아갔을 경우 어깨-손목 각도로 점수 산출
            u_i_arm = self.get_line_angle(
                [u_i_row['right_shoulder_x'], u_i_row['right_shoulder_y']],
                [u_i_row['right_wrist_x'], u_i_row['right_wrist_y']]
            )
            s3 = max(0, 100 - abs(u_i_arm - self.gt_avg['Impact_Arm_Angle']) * 2)
            metric_msg = f"임팩트 팔 각도 (User: {u_i_arm:.1f}° / GT_Avg: {self.gt_avg['Impact_Arm_Angle']:.1f}°)"

        results.append({
            '키프레임 이름': 'impact',
            '점수': round(s3, 1),
            '측정지표': metric_msg
        })

        # --- CSV 저장 및 출력 ---
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("-" * 50)
        print(f"✅ 평가 완료! 결과 저장: {output_path}")
        print(final_df)

# --- 실행부 ---
if __name__ == "__main__":
    # GT 평균값이 저장된 파일 경로
    GT_AVG_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/gt_total_metrics2.csv"
    
    # 평가할 사용자 데이터
    USER_POINTS = "/Users/minji/Documents/minton-angle_resources/roh_normalized_fixed.csv"
    USER_KF = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/roh.csv"
    
    # 결과 저장 경로
    OUTPUT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/user_evaluation2.csv"

    calculator = BadmintonScoreCalculator(GT_AVG_CSV)
    calculator.calculate_final_scores(USER_POINTS, USER_KF, OUTPUT_CSV)