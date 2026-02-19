import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path

# 1. 시각화 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class BadmintonScientificMaster:
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2)
        self.experts = ['expert_1', 'expert_2', 'expert_3', 'expert_4']
        self.expert_data = {}

    def extract_and_normalize(self, video_name):
        """영상에서 관절 추출 및 체형 정규화"""
        cap = cv2.VideoCapture(str(self.video_dir / f"{video_name}.mp4"))
        frames_data = []
        print(f"🎬 {video_name} 데이터 분석 중...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 보정 기준: 골반 중심(0,0), 척추 길이(1.0)
                hip = np.array([(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2])
                spine = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - hip) + 1e-6
                
                # 분석에 필요한 핵심 좌표만 추출
                data = {
                    'sh': (np.array([lm[12].x, lm[12].y]) - hip) / spine, # 어깨
                    'el': (np.array([lm[14].x, lm[14].y]) - hip) / spine, # 팔꿈치
                    'wr': (np.array([lm[16].x, lm[16].y]) - hip) / spine, # 손목
                    'idx': (np.array([lm[20].x, lm[20].y]) - hip) / spine, # 검지(라켓)
                    'com': hip # 원본 위치(무게중심 이동 확인용)
                }
                frames_data.append(data)
        cap.release()
        return frames_data

    def calculate_angle(self, a, b, c):
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    def run(self):
        # 1. 데이터 추출 및 CSV 저장용 리스트 생성
        raw_records = []
        for name in self.experts:
            stream = self.extract_and_normalize(name)
            self.expert_data[name] = stream
            for i, d in enumerate(stream):
                raw_records.append({'expert_id': name, 'frame': i, 'wr_x': d['wr'][0], 'wr_y': d['wr'][1]})

        pd.DataFrame(raw_records).to_csv("raw_pose_data.csv", index=False)
        print("✅ raw_pose_data.csv 저장 완료")

        # 2. DTW 속도 동기화 (expert_1 기준)
        base_wr = np.array([d['wr'] for d in self.expert_data['expert_1']])
        synced_streams = []
        for name in self.experts:
            target_wr = np.array([d['wr'] for d in self.expert_data[name]])
            _, path = fastdtw(base_wr, target_wr, dist=euclidean)
            sync_map = {p[0]: p[1] for p in path}
            synced_streams.append([self.expert_data[name][sync_map.get(t, 0)] for t in range(len(base_wr))])

        # 3. 4대 과학적 지표 시각화 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF']
        frames = np.arange(len(base_wr))

        for i, stream in enumerate(synced_streams):
            # A. 팔꿈치 각도
            angles = [self.calculate_angle(d['sh'], d['el'], d['wr']) for d in stream]
            # B. 손목 각속도 (미분)
            vel = savgol_filter(np.diff(angles, prepend=angles[0]), 7, 3)
            # C. 손목 높이 (Y)
            heights = [-d['wr'][1] for d in stream]
            # D. 무게중심 이동 (X)
            com_x = [d['com'][0] for d in stream]

            axes[0, 0].plot(frames, angles, color=colors[i], label=self.experts[i])
            axes[0, 1].plot(frames, vel, color=colors[i])
            axes[1, 0].plot(frames, heights, color=colors[i])
            axes[1, 1].plot(frames, com_x, color=colors[i])

        # 그래프 꾸미기
        axes[0, 0].set_title("① 팔꿈치 신전 각도 (펴짐 리듬)", fontsize=14)
        axes[0, 1].set_title("② 손목 스냅 각속도 (폭발력)", fontsize=14)
        axes[1, 0].set_title("③ 손목 타점 높이 (Hitting Point)", fontsize=14)
        axes[1, 1].set_title("④ 신체 무게중심 이동 (체중 이동)", fontsize=14)
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("동기화된 프레임")

        plt.suptitle("배드민턴 하이클리어 운동역학적 전문가 분석 리포트", fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("scientific_swing_report.png", dpi=200)
        plt.show()

if __name__ == "__main__":
    # ❗팀장님 PC 경로로 수정하세요
    VIDEO_PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    master = BadmintonScientificMaster(VIDEO_PATH)
    master.run()