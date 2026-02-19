import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path

class ExpertParallelSynchronizer:
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2)
        self.experts = ['expert_1', 'expert_2', 'expert_3', 'expert_4']
        # 전문가별 고유 색상 (스켈레톤 색상)
        self.colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0), (0, 165, 255)] # 노랑, 녹색, 하늘, 주황

    def get_normalized_stream(self, video_name):
        cap = cv2.VideoCapture(str(self.video_dir / f"{video_name}.mp4"))
        stream = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 체형 보정: 골반 중심 원점, 척추 길이 기준 스케일링
                hip = np.array([(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2])
                spine = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - hip) + 1e-6
                
                coords = {}
                for j in range(33):
                    pt = (np.array([lm[j].x, lm[landmark.y]]) - hip) / spine # 좌표 보정
                    coords[j] = pt
                stream.append(coords)
        cap.release()
        return stream

    def run(self):
        print("🔍 전문가 데이터 로딩 중...")
        all_streams = [self.get_normalized_stream(ex) for ex in self.experts]
        
        # 1. 속도 동기화 (DTW): expert_1 기준
        print("⏳ 전 전문가 스윙 템포 동기화 중...")
        base_wrist = np.array([s[16] for s in all_streams[0]])
        synced_indices = [list(range(len(base_wrist)))]
        
        for i in range(1, 4):
            target_wrist = np.array([s[16] for s in all_streams[i]])
            _, path = fastdtw(base_wrist, target_wrist, dist=euclidean)
            sync_map = {p[0]: p[1] for p in path}
            synced_indices.append([sync_map.get(j, 0) for j in range(len(base_wrist))])

        # 2. 4열(4-Column) 영상 생성
        col_w, h = 400, 800  # 한 칸의 너비와 높이
        total_w = col_w * 4
        out = cv2.VideoWriter("expert_4way_sync.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (total_w, h))
        
        print("🎬 4열 동기화 영상 제작 중...")
        for t in range(len(base_wrist)):
            canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
            
            for i in range(4):
                idx = synced_indices[i][t]
                skeleton = all_streams[i][min(idx, len(all_streams[i])-1)]
                color = self.colors[i]
                offset_x = i * col_w # 각 전문가의 X축 시작점
                
                # 구분선 그리기
                if i > 0:
                    cv2.line(canvas, (offset_x, 0), (offset_x, h), (50, 50, 50), 1)

                # 스켈레톤 그리기
                for conn in self.mp_pose.POSE_CONNECTIONS:
                    p1_raw, p2_raw = skeleton[conn[0]], skeleton[conn[1]]
                    # 각 열의 중앙에 배치 (scale 200)
                    p1 = (int(p1_raw[0]*200 + offset_x + col_w//2), int(p1_raw[1]*200 + h//2))
                    p2 = (int(p2_raw[0]*200 + offset_x + col_w//2), int(p2_raw[1]*200 + h//2))
                    cv2.line(canvas, p1, p2, color, 3)
                
                cv2.putText(canvas, self.experts[i], (offset_x + 20, 50), 1, 1.5, color, 2)
            
            out.write(canvas)
            
        out.release()
        print("✅ 'expert_4way_sync.mp4' 생성이 완료되었습니다!")

if __name__ == "__main__":
    PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    sync_viewer = ExpertParallelSynchronizer(PATH)
    sync_viewer.run()