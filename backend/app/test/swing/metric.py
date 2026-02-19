import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ExpertGTScientificAnalyzer:
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2)
        self.experts = ['expert_1', 'expert_2', 'expert_3', 'expert_4']
        self.colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF'] # 전문가별 고유색

    def extract_features(self, video_name):
        """영상에서 관절 좌표 추출 및 체형 정규화"""
        cap = cv2.VideoCapture(str(self.video_dir / f"{video_name}.mp4"))
        stream = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 보정: 골반 중심(0,0), 척추 길이 기준 스케일링
                hip = np.array([(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2])
                spine = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - hip) + 1e-6
                
                # 핵심 관절 정규화 좌표 저장
                coords = {
                    'sh': (np.array([lm[12].x, lm[12].y]) - hip) / spine,
                    'el': (np.array([lm[14].x, lm[14].y]) - hip) / spine,
                    'wr': (np.array([lm[16].x, lm[16].y]) - hip) / spine,
                    'hip': hip # 실제 위치 (무게중심용)
                }
                stream.append(coords)
        cap.release()
        return stream

    def calculate_angle(self, a, b, c):
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    def run_analysis(self):
        print("🔍 전문가 영상 분석 및 데이터 보정 시작...")
        expert_streams = {name: self.extract_features(name) for name in self.experts}
        
        # 1. 시간 동기화 (DTW) - expert_1 기준
        base_wr = np.array([d['wr'] for d in expert_streams['expert_1']])
        synced_data = {name: [] for name in self.experts}
        
        for name in self.experts:
            target_wr = np.array([d['wr'] for d in expert_streams[name]])
            _, path = fastdtw(base_wr, target_wr, dist=euclidean)
            sync_map = {p[0]: p[1] for p in path}
            synced_data[name] = [expert_streams[name][sync_map.get(t, 0)] for t in range(len(base_wr))]

        # 2. 시각화 리포트 생성 (4개 핵심 지표)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
        frames = np.arange(len(base_wr))
        
        # 임팩트 프레임 추정 (손목 속도가 가장 빠른 지점)
        all_elbow_angles = []

        for i, name in enumerate(self.experts):
            stream = synced_data[name]
            
            # 지표 1: 팔꿈치 각도 (신전도)
            angles = [self.calculate_angle(d['sh'], d['el'], d['wr']) for d in stream]
            all_elbow_angles.append(angles)
            
            # 지표 2: 손목 높이 (Y좌표)
            heights = [-d['wr'][1] for d in stream]
            
            # 지표 3: 손목 각속도 (스냅 스피드)
            vel = savgol_filter(np.diff(angles, prepend=angles[0]), 7, 3)
            
            # 지표 4: 무게중심(COM) X축 이동
            com_x = [d['hip'][0] for d in stream]

            # 그래프 그리기
            axes[0,0].plot(frames, angles, color=self.colors[i], label=name, alpha=0.6)
            axes[0,1].plot(frames, heights, color=self.colors[i], alpha=0.6)
            axes[1,0].plot(frames, vel, color=self.colors[i], alpha=0.6)
            axes[1,1].plot(frames, com_x, color=self.colors[i], alpha=0.6)

        # 전문가 평균 정답 범위(Shadow) 추가
        mean_angle = np.mean(all_elbow_angles, axis=0)
        std_angle = np.std(all_elbow_angles, axis=0)
        axes[0,0].fill_between(frames, mean_angle-std_angle, mean_angle+std_angle, color='gray', alpha=0.2, label='정답 범위')

        # 임팩트 구간 강조 (각속도 피크 지점)
        impact_frame = np.argmax(np.mean([np.abs(np.diff(a, prepend=a[0])) for a in all_elbow_angles], axis=0))
        for ax in axes.flat:
            ax.axvline(x=impact_frame, color='red', linestyle='--', alpha=0.5)
            ax.text(impact_frame, ax.get_ylim()[1]*0.9, ' IMPACT', color='red', fontweight='bold')

        # 제목 및 라벨링
        axes[0,0].set_title("① 팔꿈치 각도 변화 (펴짐 리듬)", fontsize=14)
        axes[0,1].set_title("② 손목 높이 변화 (타점 높이)", fontsize=14)
        axes[1,0].set_title("③ 손목 각속도 (스냅 파워)", fontsize=14)
        axes[1,1].set_title("④ 무게중심 이동 (체중 이동)", fontsize=14)
        
        plt.suptitle("배드민턴 하이클리어 전문가 4인 과학적 일관성 분석", fontsize=22, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("expert_analysis_report.png", dpi=200)
        plt.show()

if __name__ == "__main__":
    # ❗ 본인 환경의 영상 폴더 경로로 수정
    PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    analyzer = ExpertGTScientificAnalyzer(PATH)
    analyzer.run_analysis()