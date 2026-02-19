import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List

class ScoreCalculator:
    """GDR 3단계 (9개 항목) 통합 채점 엔진 - 최종 보정판"""
    def __init__(self, gt_json_path: str = None):
        if gt_json_path and os.path.exists(gt_json_path):
            with open(gt_json_path, 'r', encoding='utf-8') as f:
                self.gt = json.load(f)
            print(f"✅ [Brain] 전문가 기준 로드 완료")
        else:
            self.gt = None
            print(f"⚠️ [Brain] GT 파일 없음, 기본 임계값 사용")

    def calc_angle(self, p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0: return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (norm + 1e-6), -1.0, 1.0))))

    def evaluate_user(self, df: pd.DataFrame, keyframes: Dict) -> Dict:
        try:
            # 모든 좌표를 float으로 미리 변환하여 numpy 타입 에러 방지
            e1 = df.iloc[int(keyframes['ready'])].apply(float)
            e2 = df.iloc[int(keyframes['backswing'])].apply(float)
            e3 = df.iloc[int(keyframes['impact'])].apply(float)
        except (IndexError, KeyError, TypeError):
            return {"evaluation": [], "stage_scores": {"stage1": 0, "stage2": 0, "stage3": 0}, "total_score": 0}

        results = []

        # --- 1단계: 준비자세 (3개 항목) ---
        p1 = 1 if abs(e1['right_elbow_y'] - e1['right_shoulder_y']) < 0.15 else 0
        results.append({"id": 1, "stage": 1, "name": "팔꿈치 높이", "pass": p1, "status": "PASS" if p1 else "낮음"})

        p2 = 1 if (e1['left_shoulder_y'] - e1['left_wrist_y']) > 0 else 0
        results.append({"id": 2, "stage": 1, "name": "보조 손", "pass": p2, "status": "PASS" if p2 else "내려감"})

        p3 = 1 if abs(e1['left_shoulder_x'] - e1['right_shoulder_x']) > 0.12 else 0
        results.append({"id": 3, "stage": 1, "name": "상체 열림", "pass": p3, "status": "PASS" if p3 else "닫힘"})

        # --- 2단계: 스윙 (5개 항목) ---
        # 4. 어깨 회전 (개선: 너비 변화량의 절대값 기준)
        e1_width = abs(e1['left_shoulder_x'] - e1['right_shoulder_x'])
        e3_width = abs(e3['left_shoulder_x'] - e3['right_shoulder_x'])
        rot_val = abs(e1_width - e3_width)
        p4 = 1 if rot_val > 0.02 else 0 
        results.append({"id": 4, "stage": 2, "name": "어깨 회전", "pass": p4, "status": "PASS" if p4 else "안함"})

        # 5. 팔꿈치 L자
        ang_e2 = self.calc_angle([e2['right_shoulder_x'], e2['right_shoulder_y']], [e2['right_elbow_x'], e2['right_elbow_y']], [e2['right_wrist_x'], e2['right_wrist_y']])
        p5 = 1 if 60 <= ang_e2 <= 135 else 0
        results.append({"id": 5, "stage": 2, "name": "팔꿈치 L자", "pass": p5, "status": "PASS" if p5 else "펴짐", "value": float(round(ang_e2, 1))})

        # 6. 백스윙 깊이
        p6 = 1 if (e2['right_shoulder_x'] - e2['right_wrist_x']) > -0.05 else 0
        results.append({"id": 6, "stage": 2, "name": "백스윙 깊이", "pass": p6, "status": "PASS" if p6 else "얕음"})

        # 7. 팔 펴짐
        ang_e3 = self.calc_angle([e3['right_shoulder_x'], e3['right_shoulder_y']], [e3['right_elbow_x'], e3['right_elbow_y']], [e3['right_wrist_x'], e3['right_wrist_y']])
        p7 = 1 if ang_e3 >= 150 else 0
        results.append({"id": 7, "stage": 2, "name": "팔 펴짐", "pass": p7, "status": "PASS" if p7 else "굽힘", "value": float(round(ang_e3, 1))})

        # 8. 타점 높이
        p8 = 1 if (e3['nose_y'] - e3['right_wrist_y']) > -0.15 else 0
        results.append({"id": 8, "stage": 2, "name": "타점 높이", "pass": p8, "status": "PASS" if p8 else "낮음"})

        # --- 3단계: 팔로우스루 (방향 중립적 알고리즘) ---
        f_score = 0
        status_9 = "안함"
        # 임팩트 이후 60프레임(약 2초)까지 넉넉하게 추적
        search_range = min(int(keyframes['impact']) + 60, len(df))
        
        start_x = float(e3['right_wrist_x'])
        center_x = float(e3['nose_x'])
        started_left = start_x < center_x
        
        for i in range(int(keyframes['impact']), search_range):
            curr_x = float(df.iloc[i]['right_wrist_x'])
            curr_y = float(df.iloc[i]['right_wrist_y'])
            
            # 코(중심선)를 기준으로 반대편으로 넘어갔는지 체크
            crossed = (started_left and curr_x > center_x + 0.05) or (not started_left and curr_x < center_x - 0.05)
            
            if crossed:
                f_score = 100
                status_9 = "PASS"
                break
            elif curr_y > float(e3['right_shoulder_y']):
                f_score = 50
                status_9 = "중간"

        results.append({"id": 9, "stage": 3, "name": "팔로우", "pass": 1 if f_score >= 50 else 0, "status": status_9})

        # --- 최종 점수 합산 (표준 타입 강제 변환) ---
        s1 = int(sum(r['pass'] for r in results if r['stage'] == 1) / 3 * 100)
        s2 = int(sum(r['pass'] for r in results if r['stage'] == 2) / 5 * 100)
        s3 = int(f_score)
        
        return {
            "evaluation": results,
            "stage_scores": {"stage1": s1, "stage2": s2, "stage3": s3},
            "total_score": int((s1 + s2 + s3) / 3)
        }