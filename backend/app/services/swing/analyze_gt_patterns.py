import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import mediapipe as mp

# 1. 경로 및 설정
VIDEO_DIR = Path("./expert_videos")
OUTPUT_DIR = Path("./standard_analysis")
LABELS_PATH = Path("./keyframe_labels.csv")

class FocusedSwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)
        
    def calculate_angle(self, p1, p2, p3):
        """세 점 사이의 각도 계산 (어깨-팔꿈치-손목 등)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    def get_norm_coords(self, landmarks, w, h):
        """관절 좌표 추출 및 정규화"""
        coords = {}
        # 필요한 핵심 관절만 정의
        target_joints = {
            'nose': 0, 'l_sh': 11, 'r_sh': 12, 'l_el': 13, 'r_el': 14,
            'l_wr': 15, 'r_wr': 16, 'l_hip': 23, 'r_hip': 24
        }
        for name, idx in target_joints.items():
            lm = landmarks.landmark[idx]
            coords[f'{name}_x'] = lm.x
            coords[f'{name}_y'] = lm.y
            coords[f'{name}_z'] = lm.z
        return coords

    def run_analysis(self):
        labels = pd.read_csv(LABELS_PATH)
        expert_results = []

        for _, row in labels.iterrows():
            expert_id = row['expert_id']
            cap = cv2.VideoCapture(str(VIDEO_DIR / f"{expert_id}.mp4"))
            
            data = {}
            for phase in ['E1_ready', 'E2_backswing', 'E3_impact']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row[phase])
                ret, frame = cap.read()
                if not ret: continue
                
                res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    data[phase] = self.get_norm_coords(res.pose_landmarks, 1, 1)

            # E4 (임팩트 8프레임 후 - 손목 스냅용)
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['E3_impact'] + 8)
            ret, frame = cap.read()
            if ret:
                res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    data['E4_follow'] = self.get_norm_coords(res.pose_landmarks, 1, 1)

            # --- 핵심 지표 계산 (GDR 스타일) ---
            expert_metrics = {
                'expert_id': expert_id,
                # 1단계: 준비 (E1)
                'e1_elbow_height': data['E1_ready']['r_sh_y'] - data['E1_ready']['r_el_y'], # 양수면 팔꿈치가 위
                'e1_aux_hand_height': data['E1_ready']['l_sh_y'] - data['E1_ready']['l_wr_y'],
                'e1_sh_width': abs(data['E1_ready']['l_sh_x'] - data['E1_ready']['r_sh_x']),
                
                # 2단계: 스윙 (E2, E3)
                'e2_elbow_angle': self.calculate_angle(
                    (data['E2_backswing']['r_sh_x'], data['E2_backswing']['r_sh_y']),
                    (data['E2_backswing']['r_el_x'], data['E2_backswing']['r_el_y']),
                    (data['E2_backswing']['r_wr_x'], data['E2_backswing']['r_wr_y'])
                ),
                'e3_arm_extension': self.calculate_angle(
                    (data['E3_impact']['r_sh_x'], data['E3_impact']['r_sh_y']),
                    (data['E3_impact']['r_el_x'], data['E3_impact']['r_el_y']),
                    (data['E3_impact']['r_wr_x'], data['E3_impact']['r_wr_y'])
                ),
                'e3_impact_height': data['E3_impact']['nose_y'] - data['E3_impact']['r_wr_y'],
                
                # 3단계: 마무리 (E4)
                'e4_snap_delta': self.calculate_angle( # 손목 각도 변화 유추
                    (data['E4_follow']['r_el_x'], data['E4_follow']['r_el_y']),
                    (data['E4_follow']['r_wr_x'], data['E4_follow']['r_wr_y']),
                    (data['E4_follow']['r_wr_x'], data['E4_follow']['r_wr_y'] + 0.1)
                )
            }
            expert_results.append(expert_metrics)
            cap.release()

        # 결과 저장 및 통계 도출
        df_res = pd.DataFrame(expert_results)
        final_standard = df_res.describe().to_dict()
        
        with open(OUTPUT_DIR / "golden_standard.json", "w", encoding="utf-8") as f:
            json.dump(final_standard, f, indent=4)
        
        print("✅ 전문가 4인 핵심 패턴 분석 완료!")

if __name__ == "__main__":
    analyzer = FocusedSwingAnalyzer()
    analyzer.run_analysis()