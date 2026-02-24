import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path
import sys

class ExpertParallelSynchronizer:
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5)
        self.experts = ['expert_1', 'expert_2', 'expert_3', 'expert_4']
        # 전문가별 색상 (BGR 순서: 노랑, 초록, 하늘, 주황)
        self.colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0), (0, 165, 255)]

    def get_normalized_stream(self, video_name):
        video_path = self.video_dir / f"{video_name}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        stream = []
        print(f"🎬 {video_name} 관절 추출 중...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 체형 보정: 골반 중심(0,0), 척추 길이 기준 스케일링
                hip = np.array([(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2])
                spine = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - hip) + 1e-6
                coords = {j: (np.array([lm[j].x, lm[j].y]) - hip) / spine for j in range(33)}
                stream.append(coords)
        cap.release()
        return stream

    def run(self):
        # 1. 데이터 로드
        all_streams = []
        for ex in self.experts:
            stream = self.get_normalized_stream(ex)
            if not stream:
                print(f"❌ {ex} 영상을 읽지 못했습니다. 경로를 확인하세요.")
                return
            all_streams.append(stream)
        
        # 2. DTW 속도 동기화 (expert_1 기준)
        print("⏳ 전 전문가 스윙 템포 동기화 중 (잠시만 기다려주세요)...")
        base_wrist = np.array([s[16] for s in all_streams[0]])
        synced_indices = [list(range(len(base_wrist)))]
        
        for i in range(1, 4):
            target_wrist = np.array([s[16] for s in all_streams[i]])
            _, path = fastdtw(base_wrist, target_wrist, dist=euclidean)
            sync_map = {p[0]: p[1] for p in path}
            synced_indices.append([sync_map.get(j, 0) for j in range(len(base_wrist))])

        # 3. 4열 영상 생성
        col_w, h = 400, 800
        total_w = col_w * 4
        output_name = "expert_4way_sync_final.mp4"
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (total_w, h))
        
        print(f"🎥 영상 합성 시작: {output_name}")
        for t in range(len(base_wrist)):
            canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
            for i in range(4):
                idx = synced_indices[i][t]
                skeleton = all_streams[i][min(idx, len(all_streams[i])-1)]
                color = self.colors[i]
                offset_x = i * col_w
                
                # 가이드 선 및 이름
                if i > 0: cv2.line(canvas, (offset_x, 0), (offset_x, h), (70, 70, 70), 1)
                cv2.putText(canvas, self.experts[i], (offset_x + 20, 50), 1, 1.5, color, 2)

                # 스켈레톤 그리기
                for conn in self.mp_pose.POSE_CONNECTIONS:
                    p1_raw, p2_raw = skeleton[conn[0]], skeleton[conn[1]]
                    p1 = (int(p1_raw[0]*200 + offset_x + col_w//2), int(p1_raw[1]*200 + h//2))
                    p2 = (int(p2_raw[0]*200 + offset_x + col_w//2), int(p2_raw[1]*200 + h//2))
                    cv2.line(canvas, p1, p2, color, 3)
            
            out.write(canvas)
            if t % 10 == 0: print(f"Processing... {t}/{len(base_wrist)}")
            
        out.release()
        print(f"✅ 완료! {Path(output_name).absolute()}")

if __name__ == "__main__":
    # ❗팀장님 폴더 경로가 맞는지 마지막으로 확인!
    PATH = r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos"
    sync_viewer = ExpertParallelSynchronizer(PATH)
    sync_viewer.run()