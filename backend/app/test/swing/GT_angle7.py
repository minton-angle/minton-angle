import pandas as pd
import numpy as np
import math
import os
import json

class BadmintonSuperStrictCalculator:
    def __init__(self, gt_total_metrics_path):
        # 1. 필터링된 GT 평균 데이터 로드
        gt_df = pd.read_csv(gt_total_metrics_path)
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

    def calculate_final_scores(self, user_points_csv, user_kf_csv, output_json_path):
        # 사용자 데이터 로드
        u_df = pd.read_csv(user_points_csv, header=None, names=self.columns).apply(pd.to_numeric, errors='coerce').dropna()
        u_kf = pd.read_csv(user_kf_csv).iloc[0].to_dict()
        
        results = []

        # --- [1] READY 점수 (계수 10000) ---
        u_r_idx = int(float(u_kf['ready']))
        u_ready_h = u_df.iloc[u_r_idx]['right_shoulder_y'] - u_df.iloc[u_r_idx]['right_elbow_y']
        err_h = abs(u_ready_h - self.gt_avg['Ready_Elbow_Height'])
        s1 = max(0, 100 - err_h * 10000)
        results.append({'단계': 'ready', '점수': round(float(s1), 1), '부여 사유': f"고도 오차 {err_h:.4f}"})

        # --- [2] BACKSWING 점수 (계수 200) ---
        u_b_idx = int(float(u_kf['backswing']))
        u_b_row = u_df.iloc[u_b_idx]
        u_bs_ang = self.get_angle_3pt([u_b_row['right_shoulder_x'], u_b_row['right_shoulder_y']],
                                      [u_b_row['right_elbow_x'], u_b_row['right_elbow_y']],
                                      [u_b_row['right_wrist_x'], u_b_row['right_wrist_y']])
        err_bs = abs(u_bs_ang - self.gt_avg['Backswing_Angle'])
        s2 = max(0, 100 - err_bs * 200)
        results.append({'단계': 'backswing', '점수': round(float(s2), 1), '부여 사유': f"각도 오차 {err_bs:.2f}도"})

        # --- [3] IMPACT 점수 (계수 200) ---
        u_i_idx = int(float(u_kf['impact']))
        u_i_row = u_df.iloc[u_i_idx]
        u_bs_hip = self.get_line_angle([u_b_row['left_hip_x'], u_b_row['left_hip_y']], [u_b_row['right_hip_x'], u_b_row['right_hip_y']])
        u_i_hip = self.get_line_angle([u_i_row['left_hip_x'], u_i_row['left_hip_y']], [u_i_row['right_hip_x'], u_i_row['right_hip_y']])
        u_rot_delta = abs(u_i_hip - u_bs_hip)
        
        rotation_threshold = self.gt_avg['Impact_Rotation_Delta'] * 0.3
        
        if u_rot_delta < rotation_threshold:
            s3 = 0.0
            reason3 = f"골반 회전 부족 ({u_rot_delta:.2f}도)"
        else:
            u_i_arm = self.get_line_angle([u_i_row['right_shoulder_x'], u_i_row['right_shoulder_y']], [u_i_row['right_wrist_x'], u_i_row['right_wrist_y']])
            err_i = abs(u_i_arm - self.gt_avg['Impact_Arm_Angle'])
            s3 = max(0, 100 - err_i * 200)
            reason3 = f"팔 각도 오차 {err_i:.2f}도"
        results.append({'단계': 'impact', '점수': round(float(s3), 1), '부여 사유': reason3})

        # --- 평균 점수 산출 ---
        overall_avg = round(sum(item['점수'] for item in results) / len(results), 1)

        # --- JSON 데이터 구조화 ---
        final_data = {
            "user_evaluation": results,
            "overall_average": overall_avg,
            "standard_used": "FILTERED_AVERAGE"
        }

        # JSON 파일 저장
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ JSON 리포트 저장 완료: {output_json_path}")

# --- 실행부 ---
if __name__ == "__main__":
    calc = BadmintonSuperStrictCalculator("/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/gt_total_metrics2.csv")
    calc.calculate_final_scores(
        "/Users/minji/Documents/minton-angle_resources/roh2_normalized_fixed.csv",
        "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/roh2.csv",
        "/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/user_evaluation_roh2_1.json"
    )