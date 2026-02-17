"""
전문가 GT 범위 추출 (10개 평가 항목)

정규화: bbox 기준 (체형/키/위치 통일)
출력:
  - gt_range.json: 전문가 범위 (Golden Standard)
  - gt_metrics.csv: 전문가별 개별 값
  - viz_boxplot.png: 박스플롯 시각화
  - viz_keyframes.png: 키프레임 비교 (검정 배경)
  - viz_skeleton.png: 스켈레톤 오버레이
  - viz_radar.png: 레이더 차트

실행: python extract_gt.py
"""

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mediapipe as mp

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
VIDEO_DIR = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos")
OUTPUT_DIR = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard")
LABELS_PATH = OUTPUT_DIR / "keyframe_labels.csv"

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class GTExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.all_metrics = []
        self.all_frames = {}  # expert_id -> {E1: frame, E2: frame, E3: frame}
        self.all_poses = {}   # expert_id -> {E1: landmarks, E2: landmarks, E3: landmarks}
        
    def run(self):
        """전체 실행"""
        
        print("=" * 70)
        print("📊 전문가 GT 범위 추출 (10개 평가 항목)")
        print("=" * 70)
        
        # 1. 레이블 로드
        labels = pd.read_csv(LABELS_PATH)
        print(f"✅ 레이블 로드: {len(labels)}명\n")
        
        # 2. 각 전문가별 처리
        for _, row in labels.iterrows():
            expert_id = row['expert_id']
            e1_idx = int(row['E1_ready'])
            e2_idx = int(row['E2_backswing'])
            e3_idx = int(row['E3_impact'])
            
            print(f"🎬 {expert_id} (E1={e1_idx}, E2={e2_idx}, E3={e3_idx})...", end=" ")
            
            result = self.process_expert(expert_id, e1_idx, e2_idx, e3_idx)
            if result:
                self.all_metrics.append(result)
                print("✅")
            else:
                print("❌")
        
        # 3. GT 범위 계산
        gt_range = self.calculate_gt_range()
        
        # 4. 저장
        self.save_results(gt_range)
        
        # 5. 시각화
        self.visualize_all(gt_range)
        
        print("\n" + "=" * 70)
        print("✅ 완료!")
        print(f"   📁 출력 폴더: {OUTPUT_DIR}")
        print("=" * 70)
    
    def process_expert(self, expert_id, e1_idx, e2_idx, e3_idx):
        """단일 전문가 처리"""
        
        # 영상 경로
        video_path = VIDEO_DIR / f"{expert_id}.mp4"
        if not video_path.exists():
            video_path = VIDEO_DIR / f"{expert_id}.MP4"
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 키프레임 추출 & 포즈 추출
        frames = {}
        poses = {}
        
        for name, idx in [('E1', e1_idx), ('E2', e2_idx), ('E3', e3_idx)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            
            frames[name] = frame
            
            # MediaPipe 포즈 추출
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            
            if not result.pose_landmarks:
                cap.release()
                return None
            
            poses[name] = self.extract_landmarks(result.pose_landmarks, frame.shape)
        
        # 팔로우스루용: E3 이후 프레임들
        post_e3_poses = []
        for idx in range(e3_idx, min(e3_idx + 30, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            if result.pose_landmarks:
                post_e3_poses.append(self.extract_landmarks(result.pose_landmarks, frame.shape))
        
        cap.release()
        
        # 저장
        self.all_frames[expert_id] = frames
        self.all_poses[expert_id] = poses
        
        # bbox 정규화
        e1 = self.normalize_bbox(poses['E1'])
        e2 = self.normalize_bbox(poses['E2'])
        e3 = self.normalize_bbox(poses['E3'])
        post_e3_norm = [self.normalize_bbox(p) for p in post_e3_poses]
        
        # 10개 지표 계산
        metrics = self.calculate_metrics(expert_id, e1, e2, e3, post_e3_norm)
        
        return metrics
    
    def extract_landmarks(self, landmarks, shape):
        """MediaPipe 랜드마크 → 딕셔너리"""
        h, w = shape[:2]
        
        keypoints = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        result = {}
        for name, idx in keypoints.items():
            lm = landmarks.landmark[idx]
            result[f'{name}_x'] = lm.x * w
            result[f'{name}_y'] = lm.y * h
            result[f'{name}_v'] = lm.visibility
        
        return result
    
    def normalize_bbox(self, pose):
        """bbox 기준 0~1 정규화 (체형/키/위치 통일)"""
        
        # x, y 좌표 추출
        x_vals = [v for k, v in pose.items() if k.endswith('_x')]
        y_vals = [v for k, v in pose.items() if k.endswith('_y')]
        
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        width = x_max - x_min if x_max > x_min else 1
        height = y_max - y_min if y_max > y_min else 1
        
        normalized = {}
        for k, v in pose.items():
            if k.endswith('_x'):
                normalized[k] = (v - x_min) / width
            elif k.endswith('_y'):
                normalized[k] = (v - y_min) / height
            else:
                normalized[k] = v
        
        return normalized
    
    def calc_angle(self, p1, p2, p3):
        """세 점의 각도 계산 (p2가 꼭지점)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    def calculate_metrics(self, expert_id, e1, e2, e3, post_e3):
        """10개 평가 지표 계산"""
        
        metrics = {'expert_id': expert_id}
        
        # =========================================
        # [E1] 준비자세 (2개)
        # =========================================
        
        # ① 왼손 위치: 왼손목이 왼어깨보다 얼마나 위에 있는지
        # 양수 = 왼손이 어깨보다 위
        metrics['guide_arm_height'] = float(e1['left_shoulder_y'] - e1['left_wrist_y'])
        
        # ② 상체 열림: 왼어깨X - 오른어깨X
        # 양수 = 측면 자세 (왼어깨가 오른쪽에)
        metrics['body_openness'] = float(e1['left_shoulder_x'] - e1['right_shoulder_x'])
        
        # =========================================
        # [E2] 백스윙 (2개)
        # =========================================
        
        # ③ 백스윙 깊이: 오른어깨X - 오른손목X
        # 양수 = 손목이 어깨보다 뒤
        metrics['backswing_depth'] = float(e2['right_shoulder_x'] - e2['right_wrist_x'])
        
        # ④ 팔꿈치 접힘 (L자): E2에서 팔꿈치 각도
        metrics['elbow_bend_angle'] = float(self.calc_angle(
            [e2['right_shoulder_x'], e2['right_shoulder_y']],
            [e2['right_elbow_x'], e2['right_elbow_y']],
            [e2['right_wrist_x'], e2['right_wrist_y']]
        ))
        
        # =========================================
        # [E3] 임팩트 (3개)
        # =========================================
        
        # ⑤ 팔꿈치 높이: 오른어깨Y - 오른팔꿈치Y
        # 양수 = 팔꿈치가 어깨보다 위
        metrics['elbow_height'] = float(e3['right_shoulder_y'] - e3['right_elbow_y'])
        
        # ⑥ 팔 펴짐: E3에서 팔꿈치 각도
        metrics['arm_extension_angle'] = float(self.calc_angle(
            [e3['right_shoulder_x'], e3['right_shoulder_y']],
            [e3['right_elbow_x'], e3['right_elbow_y']],
            [e3['right_wrist_x'], e3['right_wrist_y']]
        ))
        
        # ⑦ 타점 높이: 코Y - 오른손목Y
        # 양수 = 손목이 코보다 위
        metrics['hit_point_height'] = float(e3['nose_y'] - e3['right_wrist_y'])
        
        # =========================================
        # [E1→E3] 회전 (2개)
        # =========================================
        
        # ⑧ 어깨 회전량: E3 어깨차이 - E1 어깨차이
        e1_shoulder_diff = e1['left_shoulder_x'] - e1['right_shoulder_x']
        e3_shoulder_diff = e3['left_shoulder_x'] - e3['right_shoulder_x']
        metrics['shoulder_rotation'] = float(e1_shoulder_diff - e3_shoulder_diff)
        
        # ⑨ 골반 회전량: E3 골반차이 - E1 골반차이
        e1_hip_diff = e1['left_hip_x'] - e1['right_hip_x']
        e3_hip_diff = e3['left_hip_x'] - e3['right_hip_x']
        metrics['hip_rotation'] = float(e1_hip_diff - e3_hip_diff)
        
        # =========================================
        # [E3→끝] 팔로우스루 (1개)
        # =========================================
        
        # ⑩ 팔로우스루: 손목이 왼어깨를 얼마나 넘어갔는지
        max_cross = 0
        if post_e3:
            left_shoulder_x = e3['left_shoulder_x']
            for p in post_e3:
                cross = left_shoulder_x - p['right_wrist_x']
                max_cross = max(max_cross, cross)
        metrics['follow_through'] = float(max_cross)
        
        return metrics
    
    def calculate_gt_range(self):
        """GT 범위 계산"""
        
        df = pd.DataFrame(self.all_metrics)
        
        # 평가 항목 정의
        items = [
            ('guide_arm_height', '① 왼손 위치', 'E1', '> 0 이면 어깨 위'),
            ('body_openness', '② 상체 열림', 'E1', '> 0 이면 측면'),
            ('backswing_depth', '③ 백스윙 깊이', 'E2', '> 0 이면 뒤로'),
            ('elbow_bend_angle', '④ 팔꿈치 접힘', 'E2', '60~120° 적정'),
            ('elbow_height', '⑤ 팔꿈치 높이', 'E3', '> 0 이면 어깨 위'),
            ('arm_extension_angle', '⑥ 팔 펴짐', 'E3', '> 150° 적정'),
            ('hit_point_height', '⑦ 타점 높이', 'E3', '> 0 이면 머리 위'),
            ('shoulder_rotation', '⑧ 어깨 회전', 'E1→E3', '> 0 이면 회전'),
            ('hip_rotation', '⑨ 골반 회전', 'E1→E3', '> 0 이면 회전'),
            ('follow_through', '⑩ 팔로우스루', 'E3→끝', '> 0 이면 넘어감'),
        ]
        
        gt_range = {}
        
        print("\n" + "=" * 70)
        print("📊 전문가 GT 범위 (Golden Standard)")
        print("=" * 70)
        
        for key, name, phase, criteria in items:
            values = df[key].values
            
            gt_range[key] = {
                'name': name,
                'phase': phase,
                'min': round(float(np.min(values)), 4),
                'max': round(float(np.max(values)), 4),
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values)), 4),
                'values': [round(float(v), 4) for v in values],
                'criteria': criteria,
                'pass_condition': '> 0' if 'angle' not in key else f'>= {round(float(np.min(values)), 1)}'
            }
            
            print(f"\n{name} [{phase}]")
            print(f"   값: {[round(v, 3) for v in values]}")
            print(f"   범위: {gt_range[key]['min']:.3f} ~ {gt_range[key]['max']:.3f}")
            print(f"   평균: {gt_range[key]['mean']:.3f} ± {gt_range[key]['std']:.3f}")
            print(f"   기준: {criteria}")
        
        return gt_range
    
    def save_results(self, gt_range):
        """결과 저장"""
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. JSON 저장 (Golden Standard)
        json_path = OUTPUT_DIR / "gt_range.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(gt_range, f, indent=2, ensure_ascii=False)
        print(f"\n✅ gt_range.json 저장")
        
        # 2. CSV 저장 (개별 값)
        csv_path = OUTPUT_DIR / "gt_metrics.csv"
        df = pd.DataFrame(self.all_metrics)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ gt_metrics.csv 저장")
        
        # 3. 요약 CSV (발표용)
        summary_data = []
        for key, data in gt_range.items():
            summary_data.append({
                '항목': data['name'],
                '구간': data['phase'],
                '최소': data['min'],
                '최대': data['max'],
                '평균': data['mean'],
                '표준편차': data['std'],
                '판단기준': data['criteria']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_DIR / "gt_summary.csv", index=False, encoding='utf-8-sig')
        print(f"✅ gt_summary.csv 저장")
    
    def visualize_all(self, gt_range):
        """전체 시각화"""
        
        print("\n📊 시각화 생성 중...")
        
        self.viz_boxplot(gt_range)
        self.viz_keyframes()
        self.viz_skeleton()
        self.viz_radar(gt_range)
        self.viz_comparison_table(gt_range)
    
    def viz_boxplot(self, gt_range):
        """박스플롯 시각화"""
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#FF6B6B',  # E1 - 빨강
                  '#4ECDC4', '#4ECDC4',  # E2 - 청록
                  '#45B7D1', '#45B7D1', '#45B7D1',  # E3 - 파랑
                  '#96CEB4', '#96CEB4',  # 회전 - 초록
                  '#FFEAA7']  # 팔로우스루 - 노랑
        
        for i, (key, data) in enumerate(gt_range.items()):
            ax = axes[i]
            values = data['values']
            
            bp = ax.boxplot(values, patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(0.7)
            
            # 개별 점
            x = np.ones(len(values)) + np.random.normal(0, 0.05, len(values))
            ax.scatter(x, values, c='black', s=100, zorder=5, edgecolor='white', linewidth=2)
            
            # 전문가 라벨
            for j, v in enumerate(values):
                ax.annotate(f'E{j+1}', (x[j], v), fontsize=8, ha='center', va='bottom')
            
            # 0 기준선
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            
            ax.set_title(f"{data['name']}\n({data['phase']})", fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('전문가 4명 GT 범위 (10개 평가 항목)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "viz_boxplot.png", dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print("   ✅ viz_boxplot.png")
    
    def viz_keyframes(self):
        """키프레임 비교 (검정 배경)"""
        
        n_experts = len(self.all_frames)
        
        fig, axes = plt.subplots(n_experts, 3, figsize=(15, 5 * n_experts))
        fig.patch.set_facecolor('black')
        
        if n_experts == 1:
            axes = [axes]
        
        titles = ['E1: 준비자세', 'E2: 백스윙', 'E3: 임팩트']
        
        for i, (expert_id, frames) in enumerate(self.all_frames.items()):
            for j, key in enumerate(['E1', 'E2', 'E3']):
                ax = axes[i][j] if n_experts > 1 else axes[j]
                ax.set_facecolor('black')
                
                img = cv2.cvtColor(frames[key], cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                
                if i == 0:
                    ax.set_title(titles[j], fontsize=14, fontweight='bold', color='white', pad=10)
                
                if j == 0:
                    ax.set_ylabel(expert_id, fontsize=12, fontweight='bold', color='white')
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                for spine in ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(2)
        
        plt.suptitle('전문가 키프레임 비교', fontsize=18, fontweight='bold', color='white', y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "viz_keyframes.png", dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        plt.close()
        print("   ✅ viz_keyframes.png")
    
    def viz_skeleton(self):
        """스켈레톤 오버레이"""
        
        n_experts = len(self.all_frames)
        
        fig, axes = plt.subplots(n_experts, 3, figsize=(15, 5 * n_experts))
        fig.patch.set_facecolor('black')
        
        if n_experts == 1:
            axes = [axes]
        
        titles = ['E1: 준비자세', 'E2: 백스윙', 'E3: 임팩트']
        
        for i, (expert_id, frames) in enumerate(self.all_frames.items()):
            for j, key in enumerate(['E1', 'E2', 'E3']):
                ax = axes[i][j] if n_experts > 1 else axes[j]
                ax.set_facecolor('black')
                
                frame = frames[key].copy()
                
                # MediaPipe로 스켈레톤 그리기
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.pose.process(rgb)
                
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        result.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                
                if i == 0:
                    ax.set_title(titles[j], fontsize=14, fontweight='bold', color='white', pad=10)
                
                if j == 0:
                    ax.set_ylabel(expert_id, fontsize=12, fontweight='bold', color='white')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('전문가 스켈레톤 오버레이', fontsize=18, fontweight='bold', color='white', y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "viz_skeleton.png", dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        plt.close()
        print("   ✅ viz_skeleton.png")
    
    def viz_radar(self, gt_range):
        """레이더 차트 (발표용)"""
        
        df = pd.DataFrame(self.all_metrics)
        
        # 정규화 (0~1 스케일)
        normalized = {}
        for key in gt_range.keys():
            values = df[key].values
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized[key] = (values - min_val) / (max_val - min_val)
            else:
                normalized[key] = np.ones_like(values) * 0.5
        
        # 레이더 차트
        labels = [gt_range[k]['name'] for k in gt_range.keys()]
        n_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        
        for i, row in df.iterrows():
            values = [normalized[k][i] for k in gt_range.keys()]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['expert_id'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('전문가 4명 평가 항목 비교 (정규화)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "viz_radar.png", dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("   ✅ viz_radar.png")
    
    def viz_comparison_table(self, gt_range):
        """비교 테이블 이미지 (발표용)"""
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        
        # 테이블 데이터
        headers = ['항목', '구간', '전문가 범위', '평균 ± 표준편차', 'PASS 조건']
        
        rows = []
        for key, data in gt_range.items():
            rows.append([
                data['name'],
                data['phase'],
                f"{data['min']:.3f} ~ {data['max']:.3f}",
                f"{data['mean']:.3f} ± {data['std']:.3f}",
                data['criteria']
            ])
        
        colors_map = {
            'E1': '#FFE5E5',
            'E2': '#E5F5F5', 
            'E3': '#E5F0FF',
            'E1→E3': '#E5FFE5',
            'E3→끝': '#FFF5E5'
        }
        
        row_colors = [colors_map.get(gt_range[k]['phase'], 'white') for k in gt_range.keys()]
        
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['#4A4A4A'] * len(headers),
            cellColours=[[c] * len(headers) for c in row_colors]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # 헤더 텍스트 흰색
        for i in range(len(headers)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        plt.title('전문가 GT 범위 (Golden Standard)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "viz_table.png", dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("   ✅ viz_table.png")


if __name__ == "__main__":
    extractor = GTExtractor()
    extractor.run()