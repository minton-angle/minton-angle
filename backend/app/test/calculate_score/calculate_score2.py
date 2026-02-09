import pandas as pd
import numpy as np
import cv2
import os
import math

class BadmintonFinalAnalyzer:
    def __init__(self, gt_points_csv, user_points_csv, gt_kf_csv, user_kf_csv, gt_img_dir, user_img_dir):
        # 1. 좌표 데이터 컬럼명 설정 (MediaPipe 33개 포인트 x, y)
        mp_parts = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 
            'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 
            'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 
            'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 
            'right_heel', 'left_foot_index', 'right_foot_index'
        ]
        columns = []
        for part in mp_parts:
            columns.extend([f"{part}_x", f"{part}_y"])

        # 2. 좌표 데이터 로드 및 숫자 강제 변환 (TypeError 방지)
        self.gt_df = pd.read_csv(gt_points_csv, header=None, names=columns).apply(pd.to_numeric, errors='coerce')
        self.user_df = pd.read_csv(user_points_csv, header=None, names=columns).apply(pd.to_numeric, errors='coerce')
        
        # NaN 행 제거 (헤더가 섞여 들어온 경우 대비)
        self.gt_df = self.gt_df.dropna().reset_index(drop=True)
        self.user_df = self.user_df.dropna().reset_index(drop=True)

        # 3. 키프레임 CSV 로드 (첫 행 이름, 둘째 행 번호 구조)
        gt_kf_df = pd.read_csv(gt_kf_csv)
        user_kf_df = pd.read_csv(user_kf_csv)
        
        # 첫 번째 데이터 행(iloc[0])을 딕셔너리로 변환
        self.gt_kf = gt_kf_df.iloc[0].to_dict()
        self.user_kf = user_kf_df.iloc[0].to_dict()
        
        self.gt_img_dir = gt_img_dir
        self.user_img_dir = user_img_dir
        
        # 가중치 설정 (Ready/Backswing용)
        self.weights = {'elbow': 0.4, 'backswing': 0.3, 'rotation': 0.3}

    # --- [수학적 계산 유틸리티 함수] ---
    def get_angle_3pt(self, p1, p2, p3):
        """세 점 사이의 사이각 계산 (p2가 꼭짓점)"""
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba = a - b
        bc = c - b
        
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0: return 0
        
        cosine_angle = np.dot(ba, bc) / norm
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return float(angle)

    def get_line_angle(self, p1, p2):
        """두 점을 이은 선분이 수평선(지면)과 이루는 각도"""
        dx = float(p2[0]) - float(p1[0])
        dy = float(p1[1]) - float(p2[1]) # Y축 반전 (이미지 좌표계 대응)
        return abs(math.degrees(math.atan2(dy, dx)))

    # --- [골반 회전 감지 로직] ---
    def check_pelvis_rotation(self):
        """백스윙 대비 임팩트 시 골반 너비 변화 감지 (2D 투영)"""
        try:
            bs_idx = int(float(self.user_kf['backswing']))
            impact_idx = int(float(self.user_kf['impact']))
            
            bs_row = self.user_df.iloc[bs_idx]
            impact_row = self.user_df.iloc[impact_idx]
            
            # 골반 너비 계산 (X축 거리)
            bs_hip_w = abs(float(bs_row['right_hip_x']) - float(bs_row['left_hip_x']))
            impact_hip_w = abs(float(impact_row['right_hip_x']) - float(impact_row['left_hip_x']))
            
            # 임팩트 때 몸이 정면을 보면서 골반 사이 X거리가 늘어났는지 확인
            return (impact_hip_w - bs_hip_w) > 0.03
        except Exception as e:
            print(f"골반 회전 확인 중 오류: {e}")
            return False

    # --- [단계별 점수 산출 로직] ---
    def calculate_stage_score(self, stage, has_rotated):
        try:
            u_idx = int(float(self.user_kf[stage]))
            u_row = self.user_df.iloc[u_idx]
            
            # 주요 좌표 추출 및 실수형 변환
            r_sh = [float(u_row['right_shoulder_x']), float(u_row['right_shoulder_y'])]
            l_sh = [float(u_row['left_shoulder_x']), float(u_row['left_shoulder_y'])]
            r_el = [float(u_row['right_elbow_x']), float(u_row['right_elbow_y'])]
            r_wr = [float(u_row['right_wrist_x']), float(u_row['right_wrist_y'])]

            if stage == 'impact':
                if not has_rotated:
                    return 0.0, "골반 회전 부족으로 인한 임팩트 무효"
                
                # 임팩트 각도 기준: 45~75도 사이 100점
                impact_angle = self.get_line_angle(r_sh, r_wr)
                if 45 <= impact_angle <= 75:
                    score = 100.0
                else:
                    diff = min(abs(impact_angle - 45), abs(impact_angle - 75))
                    score = max(20.0, 100.0 - (diff * 2))
                return round(score, 1), f"임팩트 각도({impact_angle:.1f}도)"

            else:
                # 1. 팔꿈치 높이 (어깨Y - 팔꿈치Y)
                h_diff = r_sh[1] - r_el[1]
                s1 = 100 if h_diff >= 0.15 else (70 if h_diff >= 0.05 else 40)
                
                # 2. 백스윙 각도 (어깨-팔꿈치-손목)
                angle = self.get_angle_3pt(r_sh, r_el, r_wr)
                s2 = 100 if 60 <= angle <= 90 else (70 if 40 <= angle < 60 else 40)
                
                # 3. 몸통 회전 (양 어깨 선의 기울기)
                rot_angle = self.get_line_angle(l_sh, r_sh)
                s3 = 100 if 45 <= rot_angle <= 60 else (70 if 30 <= rot_angle < 45 else 40)
                
                total = (s1 * self.weights['elbow']) + (s2 * self.weights['backswing']) + (s3 * self.weights['rotation'])
                items = f"팔꿈치:{s1}점, 각도:{s2}점, 회전:{s3}점"
                return round(total, 1), items
        except Exception as e:
            print(f"{stage} 점수 계산 오류: {e}")
            return 0.0, "데이터 오류"

    # --- [실행 및 시각화] ---
    def run(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        has_rotated = self.check_pelvis_rotation()
        results = []

        for stage in ['ready', 'backswing', 'impact']:
            score, items = self.calculate_stage_score(stage, has_rotated)
            results.append({'단계': stage, '총점': score, '평가항목': items})

            # 이미지 합성
            gt_img_name = f"GT1_normalized_{stage}.jpg"
            user_img_name = f"roh_normalized_fixed_{stage}.jpg"
            
            img_gt = cv2.imread(os.path.join(self.gt_img_dir, gt_img_name))
            img_user = cv2.imread(os.path.join(self.user_img_dir, user_img_name))
            
            if img_gt is not None and img_user is not None:
                img_user = cv2.resize(img_user, (img_gt.shape[1], img_gt.shape[0]))
                combined = cv2.hconcat([img_gt, img_user])
                cv2.putText(combined, f"{stage.upper()}: {score}pts", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(output_dir, f"comparison_{stage}.jpg"), combined)

        # CSV 리포트 저장
        df_report = pd.DataFrame(results)
        avg_score = round(df_report['총점'].mean(), 1)
        df_report.loc[len(df_report)] = ['종합 평균', avg_score, '-']
        
        df_report.to_csv(os.path.join(output_dir, "comparison_report_final.csv"), index=False, encoding='utf-8-sig')
        print(f"✅ 분석 완료! 최종 평균 점수: {avg_score}")

# --- [메인 실행부] ---
if __name__ == "__main__":
    analyzer = BadmintonFinalAnalyzer(
        gt_points_csv = "/Users/minji/Documents/minton-angle_resources/GT1_normalized_fixed.csv",
        user_points_csv = "/Users/minji/Documents/minton-angle_resources/GT2_normalized_fixed.csv",
        gt_kf_csv = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv",
        user_kf_csv = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT2.csv",
        gt_img_dir = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1",
        user_img_dir = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT2"
    )
    
    analyzer.run(output_dir = "/Users/minji/Documents/minton-angle/backend/data/standard/calculated_score2")