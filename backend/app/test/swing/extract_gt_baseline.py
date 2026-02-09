"""
GT 동영상 4개 분석 → 평균 기준값 추출
"""

import pandas as pd
import numpy as np
import json


class GTBaselineExtractor:
    """GT 기준값 추출기"""
    
    def __init__(self):
        # GT 설정
        self.gt_configs = {
            'GT1': {
                'kf1': 27, 'kf2': 47, 'kf3': 57,
                'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT1_normalized_fixed.csv"
            },
            'GT2': {
                'kf1': 26, 'kf2': 40, 'kf3': 48,
                'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT2_normalized_fixed.csv"
            },
            'GT3': {
                'kf1': 37, 'kf2': 59, 'kf3': 66,
                'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT3_normalized_fixed.csv"
            },
            'GT4': {
                'kf1': 39, 'kf2': 52, 'kf3': 59,
                'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT4_normalized_fixed.csv"
            }
        }
    
    def calculate_angle(self, p1, p2, p3):
        """p1-p2-p3가 이루는 각도 (p2가 꼭짓점)"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / norm
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        return float(angle)
    
    def calculate_line_angle(self, p1, p2):
        """두 점을 연결한 선분의 수평선 대비 각도"""
        dx = p2[0] - p1[0]
        dy = p1[1] - p2[1]  # Y축 반전
        
        if dx == 0:
            return 90.0
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return abs(float(angle))
    
    def analyze_kf1(self, row):
        """준비자세 (KF1) 분석"""
        
        # 1. 손목-팔꿈치-어깨 각도
        r_wrist = [row['right_wrist_x'], row['right_wrist_y']]
        r_elbow = [row['right_elbow_x'], row['right_elbow_y']]
        r_shoulder = [row['right_shoulder_x'], row['right_shoulder_y']]
        
        arm_angle = self.calculate_angle(r_wrist, r_elbow, r_shoulder)
        
        # 2. 보조손 위치
        l_wrist_y = row['left_wrist_y']
        l_elbow_y = row['left_elbow_y']
        support_hand_ok = l_wrist_y < l_elbow_y
        
        return {
            'arm_triangle_angle': arm_angle,
            'support_hand_valid': support_hand_ok
        }
    
    def analyze_kf2(self, row):
        """백스윙 (KF2) 분석"""
        
        # 1. 팔꿈치 높이
        elbow_height = row['right_shoulder_y'] - row['right_elbow_y']
        
        # 2. 백스윙 각도
        r_shoulder = [row['right_shoulder_x'], row['right_shoulder_y']]
        r_elbow = [row['right_elbow_x'], row['right_elbow_y']]
        r_wrist = [row['right_wrist_x'], row['right_wrist_y']]
        
        backswing_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # 3. 몸통 회전
        l_shoulder = [row['left_shoulder_x'], row['left_shoulder_y']]
        body_rotation = self.calculate_line_angle(l_shoulder, r_shoulder)
        
        return {
            'elbow_height': elbow_height,
            'backswing_angle': backswing_angle,
            'body_rotation': body_rotation
        }
    
    def analyze_kf3(self, row):
        """임팩트 (KF3) 분석"""
        
        # 1. 팔꿈치 펴짐
        r_shoulder = [row['right_shoulder_x'], row['right_shoulder_y']]
        r_elbow = [row['right_elbow_x'], row['right_elbow_y']]
        r_wrist = [row['right_wrist_x'], row['right_wrist_y']]
        
        elbow_extension = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # 2. 타구 위치
        impact_position = row['right_wrist_x'] - row['nose_x']
        
        # 3. 임팩트 각도
        impact_angle = self.calculate_line_angle(r_shoulder, r_wrist)
        
        return {
            'elbow_extension': elbow_extension,
            'impact_position': impact_position,
            'impact_angle': impact_angle
        }
    
    def analyze_single_gt(self, gt_name):
        """단일 GT 분석"""
        
        print(f"\n📊 {gt_name} 분석 중...")
        
        config = self.gt_configs[gt_name]
        
        # CSV 로드
        df = pd.read_csv(config['csv'])
        
        # 각 키프레임 분석
        kf1_metrics = self.analyze_kf1(df.iloc[config['kf1']])
        kf2_metrics = self.analyze_kf2(df.iloc[config['kf2']])
        kf3_metrics = self.analyze_kf3(df.iloc[config['kf3']])
        
        # 결과 출력
        print(f"  KF1: {kf1_metrics}")
        print(f"  KF2: {kf2_metrics}")
        print(f"  KF3: {kf3_metrics}")
        
        return {
            'kf1': kf1_metrics,
            'kf2': kf2_metrics,
            'kf3': kf3_metrics
        }
        
    def calculate_average_baseline(self, all_metrics):
        """4개 GT 평균 계산"""
        
        print("\n" + "=" * 60)
        print("📊 GT 평균 기준값 계산")
        print("=" * 60)
        
        # KF1 평균
        kf1_angles = [m['kf1']['arm_triangle_angle'] for m in all_metrics]
        kf1_support_valid = [m['kf1']['support_hand_valid'] for m in all_metrics]
        
        # KF2 평균
        kf2_elbow_heights = [m['kf2']['elbow_height'] for m in all_metrics]
        kf2_backswing_angles = [m['kf2']['backswing_angle'] for m in all_metrics]
        kf2_body_rotations = [m['kf2']['body_rotation'] for m in all_metrics]
        
        # KF3 평균
        kf3_elbow_extensions = [m['kf3']['elbow_extension'] for m in all_metrics]
        kf3_impact_positions = [m['kf3']['impact_position'] for m in all_metrics]
        kf3_impact_angles = [m['kf3']['impact_angle'] for m in all_metrics]
        
        # 평균 기준값 (⭐ float() 변환 추가)
        baseline = {
            'kf1': {
                'arm_triangle_angle': {
                    'mean': float(np.mean(kf1_angles)),
                    'std': float(np.std(kf1_angles)),
                    'values': [float(v) for v in kf1_angles]  # ⭐ 수정
                },
                'support_hand_valid_count': int(sum(kf1_support_valid))  # ⭐ int()
            },
            'kf2': {
                'elbow_height': {
                    'mean': float(np.mean(kf2_elbow_heights)),
                    'std': float(np.std(kf2_elbow_heights)),
                    'values': [float(v) for v in kf2_elbow_heights]  # ⭐ 수정
                },
                'backswing_angle': {
                    'mean': float(np.mean(kf2_backswing_angles)),
                    'std': float(np.std(kf2_backswing_angles)),
                    'values': [float(v) for v in kf2_backswing_angles]  # ⭐ 수정
                },
                'body_rotation': {
                    'mean': float(np.mean(kf2_body_rotations)),
                    'std': float(np.std(kf2_body_rotations)),
                    'values': [float(v) for v in kf2_body_rotations]  # ⭐ 수정
                }
            },
            'kf3': {
                'elbow_extension': {
                    'mean': float(np.mean(kf3_elbow_extensions)),
                    'std': float(np.std(kf3_elbow_extensions)),
                    'values': [float(v) for v in kf3_elbow_extensions]  # ⭐ 수정
                },
                'impact_position': {
                    'mean': float(np.mean(kf3_impact_positions)),
                    'std': float(np.std(kf3_impact_positions)),
                    'values': [float(v) for v in kf3_impact_positions]  # ⭐ 수정
                },
                'impact_angle': {
                    'mean': float(np.mean(kf3_impact_angles)),
                    'std': float(np.std(kf3_impact_angles)),
                    'values': [float(v) for v in kf3_impact_angles]  # ⭐ 수정
                }
            }
        }
        
        # 결과 출력
        print("\n📈 평균 기준값:")
        print(f"\n[KF1 - 준비자세]")
        print(f"  손목-팔꿈치-어깨 각도: {baseline['kf1']['arm_triangle_angle']['mean']:.2f}° (±{baseline['kf1']['arm_triangle_angle']['std']:.2f})")
        print(f"  보조손 올바른 위치: {baseline['kf1']['support_hand_valid_count']}/4")
        
        print(f"\n[KF2 - 백스윙]")
        print(f"  팔꿈치 높이: {baseline['kf2']['elbow_height']['mean']:.4f} (±{baseline['kf2']['elbow_height']['std']:.4f})")
        print(f"  백스윙 각도: {baseline['kf2']['backswing_angle']['mean']:.2f}° (±{baseline['kf2']['backswing_angle']['std']:.2f})")
        print(f"  몸통 회전: {baseline['kf2']['body_rotation']['mean']:.2f}° (±{baseline['kf2']['body_rotation']['std']:.2f})")
        
        print(f"\n[KF3 - 임팩트]")
        print(f"  팔꿈치 펴짐: {baseline['kf3']['elbow_extension']['mean']:.2f}° (±{baseline['kf3']['elbow_extension']['std']:.2f})")
        print(f"  타구 위치: {baseline['kf3']['impact_position']['mean']:.4f} (±{baseline['kf3']['impact_position']['std']:.4f})")
        print(f"  임팩트 각도: {baseline['kf3']['impact_angle']['mean']:.2f}° (±{baseline['kf3']['impact_angle']['std']:.2f})")
        
        return baseline
    
    def run(self, output_path):
        """전체 실행"""
        
        print("=" * 60)
        print("🎯 GT 기준값 추출 시작")
        print("=" * 60)
        
        all_metrics = []
        
        # 4개 GT 분석
        for gt_name in ['GT1', 'GT2', 'GT3', 'GT4']:
            try:
                metrics = self.analyze_single_gt(gt_name)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"❌ {gt_name} 분석 실패: {e}")
                import traceback
                traceback.print_exc()
        
        if len(all_metrics) < 4:
            print(f"\n⚠️ 경고: {len(all_metrics)}개 GT만 분석됨")
        
        # 평균 계산
        baseline = self.calculate_average_baseline(all_metrics)
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 기준값 저장 완료: {output_path}")
        
        return baseline


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    extractor = GTBaselineExtractor()
    
    # 출력 경로
    output_path = r"C:\Users\User\Desktop\CV\FinalProj\data\gt_baseline.json"
    
    # 실행
    baseline = extractor.run(output_path)