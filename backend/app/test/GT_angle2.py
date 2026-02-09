import pandas as pd
import numpy as np
import math
import os

class GTMultiAnalyzer:
    def __init__(self):
        # MediaPipe 33개 키포인트 컬럼명 생성 (x, y)
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
        """세 점 사이의 사이각 계산"""
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0: return 0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))

    def get_line_angle(self, p1, p2):
        """두 점을 이은 선분이 수평선과 이루는 각도"""
        dx = float(p2[0]) - float(p1[0])
        dy = float(p1[1]) - float(p2[1]) 
        return abs(math.degrees(math.atan2(dy, dx)))

    def extract_metrics(self, points_csv, kf_csv, gt_label):
        """특정 GT 파일로부터 4가지 지표 추출"""
        try:
            df = pd.read_csv(points_csv, header=None, names=self.columns).apply(pd.to_numeric, errors='coerce').dropna()
            kf_df = pd.read_csv(kf_csv)
            kf = kf_df.iloc[0].to_dict()

            # 1. Ready: 어깨Y - 팔꿈치Y
            r_idx = int(float(kf['ready']))
            r_row = df.iloc[r_idx]
            ready_h = round(r_row['right_shoulder_y'] - r_row['right_elbow_y'], 4)

            # 2. Backswing: 어깨-팔꿈치-손목 각도
            b_idx = int(float(kf['backswing']))
            b_row = df.iloc[b_idx]
            bs_ang = round(self.get_angle_3pt(
                [b_row['right_shoulder_x'], b_row['right_shoulder_y']],
                [b_row['right_elbow_x'], b_row['right_elbow_y']],
                [b_row['right_wrist_x'], b_row['right_wrist_y']]
            ), 2)

            # 3. Impact Rotation: Backswing 대비 골반 각도 변화량
            i_idx = int(float(kf['impact']))
            i_row = df.iloc[i_idx]
            bs_hip = self.get_line_angle([b_row['left_hip_x'], b_row['left_hip_y']], [b_row['right_hip_x'], b_row['right_hip_y']])
            i_hip = self.get_line_angle([i_row['left_hip_x'], i_row['left_hip_y']], [i_row['right_hip_x'], i_row['right_hip_y']])
            rot_delta = round(abs(i_hip - bs_hip), 2)

            # 4. Impact Arm: 어깨-손목 선의 각도
            i_arm_ang = round(self.get_line_angle(
                [i_row['right_shoulder_x'], i_row['right_shoulder_y']],
                [i_row['right_wrist_x'], i_row['right_wrist_y']]
            ), 2)

            return {
                'GT_Name': gt_label,
                'Ready_Elbow_Height': ready_h,
                'Backswing_Angle': bs_ang,
                'Impact_Rotation_Delta': rot_delta,
                'Impact_Arm_Angle': i_arm_ang
            }
        except Exception as e:
            print(f"❌ {gt_label} 분석 중 오류 발생: {e}")
            return None

# --- 실행부 ---
if __name__ == "__main__":
    analyzer = GTMultiAnalyzer()
    
    # 1. GT 1~4 파일 경로 설정
    # (파일 이름이 GT1, GT2 등으로 규칙적이라고 가정했습니다)
    base_points_path = "/Users/minji/Documents/minton-angle_resources"
    base_kf_path = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames"
    
    gt_targets = ["GT1", "GT2", "GT3", "GT4"]
    all_results = []

    for label in gt_targets:
        p_csv = f"{base_points_path}/{label}_normalized_fixed.csv"
        k_csv = f"{base_kf_path}/{label}.csv"
        
        if os.path.exists(p_csv) and os.path.exists(k_csv):
            res = analyzer.extract_metrics(p_csv, k_csv, label)
            if res:
                all_results.append(res)
                print(f"✅ {label} 분석 성공")
        else:
            print(f"⚠️ {label} 파일을 찾을 수 없습니다. (경로 확인 요망)")

    # 2. 결과 처리 및 평균 계산
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 숫자 컬럼들만 선택하여 평균 계산
        mean_values = df.mean(numeric_only=True).round(4)
        mean_row = mean_values.to_dict()
        mean_row['GT_Name'] = 'AVERAGE' # 평균 행 이름 설정
        
        # 평균 행 추가
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        # 3. CSV 저장
        output_path = "/Users/minji/Documents/minton-angle/backend/data/standard/GT_angle/gt_total_metrics.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print(f"📂 통합 리포트 저장 완료: {output_path}")
        print(df) # 결과 출력