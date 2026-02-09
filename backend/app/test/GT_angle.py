import pandas as pd
import numpy as np
import math
import os

class GTMetricsExtractor:
    def __init__(self):
        # MediaPipe 33개 키포인트 컬럼명 생성
        mp_parts = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 
            'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 
            'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 
            'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 
            'right_heel', 'left_foot_index', 'right_foot_index'
        ]
        self.columns = []
        for part in mp_parts:
            self.columns.extend([f"{part}_x", f"{part}_y"])

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

    def extract_gt_values(self, points_csv, kf_csv):
        # 데이터 로드
        df = pd.read_csv(points_csv, header=None, names=self.columns).apply(pd.to_numeric, errors='coerce').dropna()
        kf_df = pd.read_csv(kf_csv)
        kf = kf_df.iloc[0].to_dict()

        # 결과 저장용 딕셔너리
        results = {'GT_Name': os.path.basename(points_csv)}

        # 1. Ready: 어깨Y - 팔꿈치Y
        r_idx = int(float(kf['ready']))
        r_row = df.iloc[r_idx]
        results['ready_elbow_height'] = round(r_row['right_shoulder_y'] - r_row['right_elbow_y'], 4)

        # 2. Backswing: 어깨-팔꿈치-손목 각도
        b_idx = int(float(kf['backswing']))
        b_row = df.iloc[b_idx]
        results['backswing_arm_angle'] = round(self.get_angle_3pt(
            [b_row['right_shoulder_x'], b_row['right_shoulder_y']],
            [b_row['right_elbow_x'], b_row['right_elbow_y']],
            [b_row['right_wrist_x'], b_row['right_wrist_y']]
        ), 2)

        # 3. Impact: 골반 회전 (Backswing 대비 Impact 변화량)
        i_idx = int(float(kf['impact']))
        i_row = df.iloc[i_idx]
        bs_hip_ang = self.get_line_angle([b_row['left_hip_x'], b_row['left_hip_y']], [b_row['right_hip_x'], b_row['right_hip_y']])
        i_hip_ang = self.get_line_angle([i_row['left_hip_x'], i_row['left_hip_y']], [i_row['right_hip_x'], i_row['right_hip_y']])
        results['impact_pelvis_rot_delta'] = round(abs(i_hip_ang - bs_hip_ang), 2)

        # 4. Impact: 어깨-손목 각도
        results['impact_shoulder_wrist_angle'] = round(self.get_line_angle(
            [i_row['right_shoulder_x'], i_row['right_shoulder_y']],
            [i_row['right_wrist_x'], i_row['right_wrist_y']]
        ), 2)

        return results

# --- 실행부 ---
if __name__ == "__main__":
    extractor = GTMetricsExtractor()
    
    # 1. 분석할 파일 리스트 (필요 시 여러 개 추가 가능)
    gt_files = [
        {
            'points': "/Users/minji/Documents/minton-angle_resources/GT1_normalized_fixed.csv",
            'kf': "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv"
        }
    ]

    all_results = []
    for gt in gt_files:
        if os.path.exists(gt['points']) and os.path.exists(gt['kf']):
            val = extractor.extract_gt_values(gt['points'], gt['kf'])
            all_results.append(val)
            print(f"✅ {val['GT_Name']} 분석 완료")

    # 2. CSV 저장
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_path = "/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/GT1_metrics_summary.csv"
        
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n📂 결과가 CSV로 저장되었습니다: {output_path}")
        print(output_df)