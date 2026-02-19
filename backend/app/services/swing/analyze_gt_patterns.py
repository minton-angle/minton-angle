import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import mediapipe as mp

class BackswingDepthAnalyzer:
    def __init__(self, output_dir="./standard_analysis"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_backswing_depth(self, landmarks):
        """
        신체 비율 기반 백스윙 깊이 계산
        원리: (손목_x - 코_x) / 어깨너비
        => 값이 클수록 손목이 몸 뒤로 깊게 빠진 것을 의미 (오른손잡이 기준)
        """
        lm = landmarks.landmark
        
        # 데이터 품질 체크 (신뢰도 50% 미만은 무시)
        if lm[0].visibility < 0.5 or lm[11].visibility < 0.5 or \
           lm[12].visibility < 0.5 or lm[16].visibility < 0.5:
            return None

        # 1. 체형 보정용 어깨 너비(Unit) 계산
        sh_l = np.array([lm[11].x, lm[11].y])
        sh_r = np.array([lm[12].x, lm[12].y])
        shoulder_dist = np.linalg.norm(sh_l - sh_r) + 1e-6

        # 2. 코(중심축) 대비 손목의 수평 거리 비율 계산
        # 오른손잡이가 카메라를 마주볼 때, 뒤로 빼는 동작은 x값이 커지는 방향입니다.
        depth_ratio = (lm[16].x - lm[0].x) / shoulder_dist
        return depth_ratio

    def run_analysis(self, video_dir, labels_csv):
        labels = pd.read_csv(labels_csv)
        all_expert_depths = []
        
        plt.figure(figsize=(12, 7))

        for _, row in labels.iterrows():
            eid = row['expert_id']
            video_path = Path(video_dir) / f"{eid}.mp4"
            if not video_path.exists(): continue

            cap = cv2.VideoCapture(str(video_path))
            print(f"🧐 {eid} 백스윙 깊이 추출 중...")
            
            raw_depths = []
            # E1(준비) ~ E3(임팩트) 구간만 추출하여 속도 보정 대상 설정
            start, end = int(row['E1_ready']), int(row['E3_impact'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            for f_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret: break
                res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    depth = self.get_backswing_depth(res.pose_landmarks)
                    if depth is not None:
                        raw_depths.append(depth)
            
            cap.release()
            if len(raw_depths) < 10: continue

            # --- 1. 속도 보정 (Temporal Normalization) ---
            # 모든 스윙을 0% ~ 100% 진행률로 통일 (100개 지점)
            raw_depths = np.array(raw_depths)
            x_old = np.linspace(0, 100, len(raw_depths))
            x_new = np.linspace(0, 100, 100)
            resampled_depth = interp1d(x_old, raw_depths, kind='cubic')(x_new)
            
            all_expert_depths.append(resampled_depth)
            plt.plot(x_new, resampled_depth, alpha=0.3, label=f'Expert {eid}')

        # --- 2. XAI의 핵심: 전문가 합의 영역(Expert Zone) 시각화 ---
        all_expert_depths = np.array(all_expert_depths)
        mean_depth = np.mean(all_expert_depths, axis=0)
        std_depth = np.std(all_expert_depths, axis=0)

        # 1.5 표준편차 범위 색칠 (이 영역이 정석의 통로입니다)
        plt.fill_between(x_new, mean_depth - 1.5*std_depth, mean_depth + 1.5*std_depth, 
                         color='gray', alpha=0.2, label='Expert Golden Zone')
        plt.plot(x_new, mean_depth, color='red', linewidth=3, label='Expert Average')

        # 그래프 꾸미기
        plt.title("XAI Analysis: Backswing Depth Flow (Wrist X-Distance / Shoulder Width)", fontsize=15)
        plt.xlabel("Swing Progress (%)", fontsize=12)
        plt.ylabel("Normalized Depth Ratio", fontsize=12)
        plt.axvline(x=50, color='blue', linestyle=':', label='Peak Backswing Point') # 보통 50% 지점이 깊이 정점
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 로컬 저장
        save_path = self.output_dir / "expert_backswing_depth_xai.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ 분석 그래프 저장 완료: {save_path}")
        plt.show()

if __name__ == "__main__":
    # r을 붙여 경로 에러 방지
    VIDEOS = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    LABELS = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\keyframe_labels.csv"

    analyzer = BackswingDepthAnalyzer()
    analyzer.run_analysis(VIDEOS, LABELS)