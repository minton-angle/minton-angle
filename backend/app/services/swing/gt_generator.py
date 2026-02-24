"""
GT Generator: 전문가 4명의 영상에서 평가 기준 자동 생성

[흐름]
1. 전문가 영상 4개 로드
2. 각 영상에서 22개 관절 추출 (눈,입,귀 제외)
3. bbox 정규화 (체형/위치 통제)
4. 키프레임(E1, E2, E3) 레이블 로드
5. 각 키프레임에서 9개 평가 항목 값 계산
6. 4명의 통계(min, max, mean, std) 산출
7. gt_evaluation.json 저장
"""

import cv2
import json
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Dict, List, Tuple
from pathlib import Path


class GTGenerator:
    """전문가 GT 기준 생성기"""
    
    def __init__(self, expert_video_dir: str, keyframe_labels_path: str):
        """
        Args:
            expert_video_dir: 전문가 영상 폴더 (expert_1.mp4, expert_2.mp4, ...)
            keyframe_labels_path: 키프레임 레이블 CSV 경로
        """
        self.expert_video_dir = Path(expert_video_dir)
        self.keyframe_labels_path = keyframe_labels_path
        
        # MediaPipe 설정
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 22개 관절 (눈,입,귀 제외)
        self.keypoint_indices = {
            0: 'nose',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index',
            21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        # 키프레임 레이블 로드
        self.keyframe_labels = self._load_keyframe_labels()
        
        print(f"✅ GTGenerator 초기화")
        print(f"   - 전문가 영상 폴더: {self.expert_video_dir}")
        print(f"   - 키프레임 레이블: {len(self.keyframe_labels)}개 영상")
    
    def _load_keyframe_labels(self) -> Dict:
        """키프레임 레이블 CSV 로드"""
        df = pd.read_csv(self.keyframe_labels_path)
        
        labels = {}
        for _, row in df.iterrows():
            expert_id = row['expert_id']
            labels[expert_id] = {
                'E1': int(row['E1_ready']),
                'E2': int(row['E2_backswing']),
                'E3': int(row['E3_impact'])
            }
        
        return labels
    
    def extract_keypoints_from_video(self, video_path: str) -> pd.DataFrame:
        """
        영상에서 22개 관절 추출 + bbox 정규화
        
        Returns:
            DataFrame: frame_id, nose_x, nose_y, ... (정규화된 좌표)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 영상 열기 실패: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_data = []
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 원시 좌표 추출
                raw_kps = {}
                for idx, name in self.keypoint_indices.items():
                    lm = landmarks[idx]
                    raw_kps[f'{name}_x'] = lm.x
                    raw_kps[f'{name}_y'] = lm.y
                    raw_kps[f'{name}_z'] = lm.z
                    raw_kps[f'{name}_v'] = lm.visibility
                
                # bbox 정규화
                normalized = self._normalize_bbox(raw_kps)
                
                row = {'frame_id': frame_id, 'timestamp': frame_id / fps}
                row.update(normalized)
                all_data.append(row)
            
            frame_id += 1
        
        cap.release()
        
        df = pd.DataFrame(all_data)
        print(f"   ✅ {Path(video_path).name}: {len(df)}프레임 추출")
        
        return df
    
    def _normalize_bbox(self, kps: Dict) -> Dict:
        """
        bbox 기반 정규화 (체형/위치 통제)
        
        모든 관절을 0~1 범위로 정규화
        """
        # 모든 x, y 좌표 수집
        xs = [kps[f'{name}_x'] for name in self.keypoint_indices.values() if kps.get(f'{name}_v', 0) > 0.5]
        ys = [kps[f'{name}_y'] for name in self.keypoint_indices.values() if kps.get(f'{name}_v', 0) > 0.5]
        
        if not xs or not ys:
            return kps
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width < 0.01:
            width = 0.1
        if height < 0.01:
            height = 0.1
        
        # 정규화
        normalized = {}
        for name in self.keypoint_indices.values():
            x = kps[f'{name}_x']
            y = kps[f'{name}_y']
            
            normalized[f'{name}_x'] = (x - min_x) / width
            normalized[f'{name}_y'] = (y - min_y) / height
            normalized[f'{name}_z'] = kps[f'{name}_z']
            normalized[f'{name}_v'] = kps[f'{name}_v']
        
        return normalized
    
    def calc_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """3점 각도 계산 (p2가 꼭지점)"""
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
    
    def calculate_metrics_at_keyframe(
        self, 
        df: pd.DataFrame, 
        keyframes: Dict
    ) -> Dict:
        """
        키프레임에서 9개 평가 항목 값 계산
        
        Args:
            df: 정규화된 keypoints DataFrame
            keyframes: {'E1': 30, 'E2': 48, 'E3': 60}
            
        Returns:
            {
                'stage1': {
                    'elbow_height': 0.05,
                    'guide_arm': 0.12,
                    'body_open': 0.08
                },
                'stage2': {...},
                'stage3': {...}
            }
        """
        e1_idx = keyframes['E1']
        e2_idx = keyframes['E2']
        e3_idx = keyframes['E3']
        
        e1 = df.iloc[e1_idx]
        e2 = df.iloc[e2_idx]
        e3 = df.iloc[e3_idx]
        
        metrics = {
            'stage1': {},
            'stage2': {},
            'stage3': {}
        }
        
        # ═══════════════════════════════════════
        # 1단계: 준비자세 (E1)
        # ═══════════════════════════════════════
        
        # ① 팔꿈치 높이: |elbow_y - shoulder_y|
        metrics['stage1']['elbow_height'] = abs(
            e1['right_elbow_y'] - e1['right_shoulder_y']
        )
        
        # ② 보조 손: left_shoulder_y - left_wrist_y (양수면 올라감)
        metrics['stage1']['guide_arm'] = (
            e1['left_shoulder_y'] - e1['left_wrist_y']
        )
        
        # ③ 상체 열림: left_shoulder_x - right_shoulder_x (양수면 열림)
        metrics['stage1']['body_open'] = (
            e1['left_shoulder_x'] - e1['right_shoulder_x']
        )
        
        # ═══════════════════════════════════════
        # 2단계: 스윙 (E1→E3 + E2, E3)
        # ═══════════════════════════════════════
        
        # ④ 어깨 회전: E1과 E3의 어깨 차이 변화
        e1_shoulder_diff = e1['left_shoulder_x'] - e1['right_shoulder_x']
        e3_shoulder_diff = e3['left_shoulder_x'] - e3['right_shoulder_x']
        metrics['stage2']['shoulder_rotation'] = e1_shoulder_diff - e3_shoulder_diff
        
        # ⑤ 팔꿈치 L자 (E2): 어깨-팔꿈치-손목 각도
        metrics['stage2']['elbow_bend'] = self.calc_angle(
            (e2['right_shoulder_x'], e2['right_shoulder_y']),
            (e2['right_elbow_x'], e2['right_elbow_y']),
            (e2['right_wrist_x'], e2['right_wrist_y'])
        )
        
        # ⑥ 백스윙 깊이 (E2): shoulder_x - wrist_x (양수면 뒤로 감)
        metrics['stage2']['backswing_depth'] = (
            e2['right_shoulder_x'] - e2['right_wrist_x']
        )
        
        # ⑦ 팔 펴짐 (E3): 어깨-팔꿈치-손목 각도
        metrics['stage2']['arm_extension'] = self.calc_angle(
            (e3['right_shoulder_x'], e3['right_shoulder_y']),
            (e3['right_elbow_x'], e3['right_elbow_y']),
            (e3['right_wrist_x'], e3['right_wrist_y'])
        )
        
        # ⑧ 타점 높이 (E3): nose_y - wrist_y (양수면 머리 위)
        metrics['stage2']['hit_height'] = (
            e3['nose_y'] - e3['right_wrist_y']
        )
        
        # ═══════════════════════════════════════
        # 3단계: 팔로우스루 (E3 이후)
        # ═══════════════════════════════════════
        
        # ⑨ 팔로우스루: E3 이후 손목이 왼어깨를 지나갔는지
        has_followthrough = False
        left_shoulder_x = e3['left_shoulder_x']
        
        for i in range(e3_idx, min(e3_idx + 20, len(df))):
            row = df.iloc[i]
            if row['right_wrist_x'] < left_shoulder_x:
                has_followthrough = True
                break
        
        metrics['stage3']['follow_through'] = 1 if has_followthrough else 0
        
        return metrics
    
    def generate_gt(self, output_path: str):
        """
        전문가 4명의 GT 기준 생성 및 JSON 저장
        
        Args:
            output_path: 출력 JSON 경로
        """
        print(f"\n{'='*60}")
        print(f"🚀 GT 생성 시작")
        print(f"{'='*60}")
        
        # 모든 전문가의 메트릭 수집
        all_metrics = {
            'stage1': {'elbow_height': [], 'guide_arm': [], 'body_open': []},
            'stage2': {'shoulder_rotation': [], 'elbow_bend': [], 'backswing_depth': [], 
                      'arm_extension': [], 'hit_height': []},
            'stage3': {'follow_through': []}
        }
        
        expert_data = {}  # 개별 전문가 데이터 저장
        
        # 전문가 영상 처리
        for expert_id, keyframes in self.keyframe_labels.items():
            video_path = self.expert_video_dir / f"{expert_id}.mp4"
            
            if not video_path.exists():
                print(f"⚠️ 영상 없음: {video_path}")
                continue
            
            print(f"\n📹 {expert_id} 처리 중...")
            
            # keypoints 추출
            df = self.extract_keypoints_from_video(str(video_path))
            
            if df is None or len(df) == 0:
                continue
            
            # 메트릭 계산
            metrics = self.calculate_metrics_at_keyframe(df, keyframes)
            
            expert_data[expert_id] = metrics
            
            # 수집
            for stage, items in metrics.items():
                for metric_name, value in items.items():
                    all_metrics[stage][metric_name].append(value)
            
            print(f"   ✅ 메트릭 계산 완료")
        
        # 통계 계산
        print(f"\n📊 통계 계산 중...")
        
        gt_result = {
            "version": "1.0",
            "description": "배드민턴 스윙 평가 GT 기준 (전문가 4명 기반)",
            "n_experts": len(expert_data),
            "generated_from": list(expert_data.keys()),
            
            "stages": {
                "stage1": {
                    "name": "준비자세",
                    "phase": "E1",
                    "max_score": 100,
                    "metrics": {}
                },
                "stage2": {
                    "name": "스윙",
                    "phase": "E1→E3, E2, E3",
                    "max_score": 100,
                    "metrics": {}
                },
                "stage3": {
                    "name": "팔로우스루",
                    "phase": "E3→끝",
                    "max_score": 100,
                    "metrics": {}
                }
            }
        }
        
        # 메트릭 정의
        metric_definitions = {
            'stage1': {
                'elbow_height': {
                    'id': 1,
                    'name': '팔꿈치 높이',
                    'type': 'direction',
                    'description': '팔꿈치가 어깨 높이 근처에 있는지',
                    'pass_condition': 'abs(value) < threshold',
                    'fail_display': ['낮음', '높음']
                },
                'guide_arm': {
                    'id': 2,
                    'name': '보조 손',
                    'type': 'direction',
                    'description': '왼손이 위로 올라가 있는지',
                    'pass_condition': 'value > 0',
                    'fail_display': ['내려감']
                },
                'body_open': {
                    'id': 3,
                    'name': '상체 열림',
                    'type': 'direction',
                    'description': '상체(어깨)가 열려있는지',
                    'pass_condition': 'value > threshold',
                    'fail_display': ['닫힘']
                }
            },
            'stage2': {
                'shoulder_rotation': {
                    'id': 4,
                    'name': '어깨 회전',
                    'type': 'direction',
                    'description': 'E1→E3 동안 어깨가 회전했는지',
                    'pass_condition': 'value > threshold',
                    'fail_display': ['안함']
                },
                'elbow_bend': {
                    'id': 5,
                    'name': '팔꿈치 L자',
                    'type': 'gt_range',
                    'description': '백스윙 시 팔꿈치 각도 (L자 형태)',
                    'pass_condition': 'gt_min <= value <= gt_max',
                    'fail_display': ['펴짐', '굽힘']
                },
                'backswing_depth': {
                    'id': 6,
                    'name': '백스윙 깊이',
                    'type': 'direction',
                    'description': '손목이 어깨 뒤로 갔는지',
                    'pass_condition': 'value > 0',
                    'fail_display': ['얕음']
                },
                'arm_extension': {
                    'id': 7,
                    'name': '팔 펴짐',
                    'type': 'gt_range',
                    'description': '임팩트 시 팔꿈치 각도',
                    'pass_condition': 'value >= gt_min',
                    'fail_display': ['굽힘']
                },
                'hit_height': {
                    'id': 8,
                    'name': '타점 높이',
                    'type': 'direction',
                    'description': '손목이 머리 위에 있는지',
                    'pass_condition': 'value > 0',
                    'fail_display': ['낮음']
                }
            },
            'stage3': {
                'follow_through': {
                    'id': 9,
                    'name': '팔로우스루',
                    'type': 'binary',
                    'description': '팔로우스루를 했는지',
                    'pass_condition': 'value == 1',
                    'fail_display': ['안함']
                }
            }
        }
        
        # 통계 + 정의 결합
        for stage, metrics in all_metrics.items():
            n_metrics = len(metrics)
            score_per_metric = round(100 / n_metrics, 1)
            
            for metric_name, values in metrics.items():
                if not values:
                    continue
                
                values = np.array(values)
                
                definition = metric_definitions[stage][metric_name].copy()
                definition['score'] = score_per_metric
                definition['expert_values'] = [round(v, 4) for v in values.tolist()]
                definition['statistics'] = {
                    'min': round(float(values.min()), 4),
                    'max': round(float(values.max()), 4),
                    'mean': round(float(values.mean()), 4),
                    'std': round(float(values.std()), 4)
                }
                
                # GT 범위 타입은 min/max를 gt_min/gt_max로
                if definition['type'] == 'gt_range':
                    definition['gt_min'] = round(float(values.min()) - values.std(), 1)
                    definition['gt_max'] = round(float(values.max()) + values.std(), 1)
                
                # 방향성 타입은 threshold 설정
                elif definition['type'] == 'direction':
                    # 평균의 50%를 threshold로
                    mean_val = float(values.mean())
                    if metric_name == 'elbow_height':
                        definition['threshold'] = round(abs(mean_val) * 2, 4)  # 여유있게
                    else:
                        definition['threshold'] = round(mean_val * 0.5, 4)
                
                gt_result['stages'][stage]['metrics'][metric_name] = definition
        
        # 개별 전문가 데이터 추가
        gt_result['expert_raw_data'] = {}
        for expert_id, metrics in expert_data.items():
            gt_result['expert_raw_data'][expert_id] = {
                stage: {k: round(v, 4) for k, v in items.items()}
                for stage, items in metrics.items()
            }
        
        # JSON 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(gt_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ GT 생성 완료!")
        print(f"   - 저장 경로: {output_path}")
        print(f"   - 전문가 수: {len(expert_data)}명")
        print(f"   - 평가 항목: 9개")
        print(f"{'='*60}")
        
        # 요약 출력
        print(f"\n📋 GT 요약:")
        for stage, data in gt_result['stages'].items():
            print(f"\n  [{data['name']}] ({data['phase']})")
            for metric_name, metric in data['metrics'].items():
                stats = metric['statistics']
                if metric['type'] == 'gt_range':
                    print(f"    - {metric['name']}: {metric['gt_min']:.1f}° ~ {metric['gt_max']:.1f}° (mean: {stats['mean']:.1f}°)")
                else:
                    print(f"    - {metric['name']}: mean={stats['mean']:.4f}, threshold={metric.get('threshold', 'N/A')}")
        
        return gt_result
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


# ═══════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 경로 설정
    EXPERT_VIDEO_DIR = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    KEYFRAME_LABELS_PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\keyframe_labels.csv"
    OUTPUT_PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\gt_evaluation.json"
    
    # GT 생성
    generator = GTGenerator(EXPERT_VIDEO_DIR, KEYFRAME_LABELS_PATH)
    gt_result = generator.generate_gt(OUTPUT_PATH)