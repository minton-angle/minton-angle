"""
오버레이 생성: 원본 영상 + 실시간 분석 → 이미지/비디오 출력
"""

import pandas as pd
import numpy as np
import cv2
import os
import math
import mediapipe as mp


class OverlayGenerator:
    """오버레이 이미지/비디오 생성 (실시간 MediaPipe 재추출)"""
    
    def __init__(self):
        # MediaPipe 설정
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def get_realtime_landmarks(self, frame):
        """
        실시간 좌표 추출
        
        Args:
            frame: OpenCV 프레임
            
        Returns:
            {'nose': (x_px, y_px), 'right_shoulder': (x_px, y_px), ...}
        """
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(img_rgb)
        
        kps = {}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            mapping = {
                self.mp_pose.PoseLandmark.NOSE: 'nose',
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
                self.mp_pose.PoseLandmark.LEFT_ELBOW: 'left_elbow',
                self.mp_pose.PoseLandmark.LEFT_WRIST: 'left_wrist',
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: 'right_elbow',
                self.mp_pose.PoseLandmark.RIGHT_WRIST: 'right_wrist',
                self.mp_pose.PoseLandmark.RIGHT_PINKY: 'right_pinky',
                self.mp_pose.PoseLandmark.LEFT_HIP: 'left_hip',
                self.mp_pose.PoseLandmark.RIGHT_HIP: 'right_hip',
                self.mp_pose.PoseLandmark.LEFT_KNEE: 'left_knee',
                self.mp_pose.PoseLandmark.LEFT_ANKLE: 'left_ankle',
                self.mp_pose.PoseLandmark.RIGHT_KNEE: 'right_knee',
                self.mp_pose.PoseLandmark.RIGHT_ANKLE: 'right_ankle',
                self.mp_pose.PoseLandmark.LEFT_HEEL: 'left_heel',
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX: 'left_foot_index',
                self.mp_pose.PoseLandmark.RIGHT_HEEL: 'right_heel',
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: 'right_foot_index'
            }
            
            for mp_id, name in mapping.items():
                lm = landmarks[mp_id]
                if lm.visibility > 0.5:
                    kps[name] = (int(lm.x * w), int(lm.y * h))
        
        return kps
    
    def draw_skeleton(self, img, kps, color=(0, 255, 0), thickness=2):
        """
        스켈레톤 그리기
        
        Args:
            img: 이미지
            kps: {'nose': (x, y), ...}
            color: 색상 (B, G, R)
            thickness: 선 두께
        """
        # 관절점 그리기
        for pt in kps.values():
            cv2.circle(img, pt, 4, color, -1)
        
        # 연결선 정의
        connections_str = [
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_ankle', 'left_heel'), ('left_heel', 'left_foot_index'),
            ('left_ankle', 'left_foot_index'),
            ('right_ankle', 'right_heel'), ('right_heel', 'right_foot_index'),
            ('right_ankle', 'right_foot_index')
        ]
        
        # 선 그리기
        for p1, p2 in connections_str:
            if p1 in kps and p2 in kps:
                cv2.line(img, kps[p1], kps[p2], color, thickness)
    
    def calculate_angle_3points(self, a, b, c):
        """3점 각도 계산"""
        if not (a and b and c):
            return 0.0
        
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    def process_ready(self, frame, save_path):
        """Ready 이미지 생성"""
        overlay = frame.copy()
        kps = self.get_realtime_landmarks(frame)
        
        rw = kps.get('right_wrist')
        rp = kps.get('right_pinky')
        re = kps.get('right_elbow')
        lw = kps.get('left_wrist')
        
        angle_val = 0.0
        
        if rw and rp and re and lw:
            vec_x = rp[0] - rw[0]
            vec_y = rp[1] - rw[1]
            racket_tip = (int(rw[0] + vec_x * 4), int(rw[1] + vec_y * 4))
            
            angle_val = self.calculate_angle_3points(racket_tip, re, lw)
            
            # 삼각형 선
            cv2.line(overlay, racket_tip, re, (0, 255, 255), 3)
            cv2.line(overlay, re, lw, (0, 255, 255), 3)
            cv2.line(overlay, lw, racket_tip, (0, 255, 255), 2)
            
            # 꼭짓점 강조
            cv2.circle(overlay, racket_tip, 8, (0, 0, 255), -1)
            cv2.circle(overlay, re, 6, (0, 255, 255), -1)
            cv2.circle(overlay, lw, 6, (0, 255, 255), -1)
        
        cv2.putText(
            overlay,
            f"Ready Angle: {angle_val:.1f}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )
        
        cv2.imwrite(save_path, overlay)
        print(f"  ✅ Ready 이미지 저장")
        return {'ready_angle': round(angle_val, 2)}
    
    def process_backswing(self, frame, save_path):
        """Backswing 이미지 생성"""
        overlay = frame.copy()
        kps = self.get_realtime_landmarks(frame)
        
        self.draw_skeleton(overlay, kps, color=(255, 255, 255))
        
        nose = kps.get('nose')
        wrist = kps.get('right_wrist')
        is_behind = False
        
        if nose and wrist:
            cv2.circle(overlay, nose, 10, (0, 0, 255), -1)
            cv2.circle(overlay, wrist, 10, (0, 0, 255), -1)
            
            if wrist[0] < nose[0]:
                is_behind = True
            
            msg = "Hand Behind" if is_behind else "Hand Forward"
            col = (0, 255, 0) if is_behind else (0, 0, 255)
            
            cv2.putText(
                overlay,
                msg,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                col,
                3
            )
        
        cv2.imwrite(save_path, overlay)
        print(f"  ✅ Backswing 이미지 저장")
    
    def process_impact(self, frame, save_path):
        """Impact 이미지 생성"""
        overlay = frame.copy()
        kps = self.get_realtime_landmarks(frame)
        
        self.draw_skeleton(overlay, kps, color=(255, 255, 255))
        
        s = kps.get('right_shoulder')
        e = kps.get('right_elbow')
        w_pt = kps.get('right_wrist')
        
        if s and e and w_pt:
            # 어깨-손목 (보라)
            cv2.line(overlay, s, w_pt, (255, 0, 255), 3)
            
            # 어깨-팔꿈치-손목 (노랑)
            cv2.line(overlay, s, e, (0, 255, 255), 3)
            cv2.line(overlay, e, w_pt, (0, 255, 255), 3)
            
            # 팔꿈치 각도
            elbow_ang = 180 - self.calculate_angle_3points(s, e, w_pt)
            
            # 공격각
            dy = abs(s[1] - w_pt[1])
            dx = abs(s[0] - w_pt[0])
            att_ang = math.degrees(math.atan2(dy, dx))
            
            cv2.putText(
                overlay,
                f"Attack: {att_ang:.1f}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2
            )
            
            cv2.putText(
                overlay,
                f"Ext: {elbow_ang:.1f}",
                (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
        
        cv2.imwrite(save_path, overlay)
        print(f"  ✅ Impact 이미지 저장")
    
    def process_rotation_video(self, cap, start_f, end_f, w, h, save_path):
        """Rotation 비디오 생성 (ffmpeg 사용)"""
        import subprocess
        import tempfile
        import shutil
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        temp_dir = tempfile.mkdtemp()
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            frame_idx = 0
            for _ in range(start_f, end_f + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                kps = self.get_realtime_landmarks(frame)
                self.draw_skeleton(frame, kps, color=(200, 200, 200), thickness=2)
                
                rs = kps.get('right_shoulder')
                ls = kps.get('left_shoulder')
                rh = kps.get('right_hip')
                lh = kps.get('left_hip')
                
                if rs and ls:
                    cv2.line(frame, rs, ls, (0, 0, 255), 5)
                
                if rh and lh:
                    cv2.line(frame, rh, lh, (255, 0, 0), 5)
                
                cv2.putText(
                    frame,
                    "Rotation Analysis",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                temp_img_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(temp_img_path, frame)
                frame_idx += 1
            
            # ffmpeg 실행
            cmd = [
                'ffmpeg',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                save_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ ffmpeg 에러: {result.stderr}")
            else:
                print(f"  ✅ Rotation 비디오 생성")
        
        finally:
            shutil.rmtree(temp_dir)
    
    def process_follow_video(self, cap, start_f, w, h, save_path):
        """Follow 비디오 생성 (ffmpeg 사용)"""
        import subprocess
        import tempfile
        import shutil
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        temp_dir = tempfile.mkdtemp()
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            traj_points = []
            frame_idx = 0
            
            for _ in range(25):
                ret, frame = cap.read()
                if not ret:
                    break
                
                kps = self.get_realtime_landmarks(frame)
                self.draw_skeleton(frame, kps, color=(255, 255, 255))
                
                wrist = kps.get('right_wrist')
                if wrist:
                    traj_points.append(wrist)
                    cv2.circle(frame, wrist, 10, (0, 0, 255), -1)
                
                if len(traj_points) > 1:
                    cv2.polylines(
                        frame,
                        [np.array(traj_points)],
                        False,
                        (0, 0, 255),
                        3
                    )
                
                temp_img_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(temp_img_path, frame)
                frame_idx += 1
            
            # ffmpeg 실행
            cmd = [
                'ffmpeg',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                save_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ ffmpeg 에러: {result.stderr}")
            else:
                print(f"  ✅ Follow 비디오 생성")
        
        finally:
            shutil.rmtree(temp_dir)
    
    def generate_all_outputs(
        self,
        video_path: str,
        keyframes: dict,
        output_dir: str
    ):
        """
        모든 오버레이 생성
        
        Args:
            video_path: 원본 비디오 경로
            keyframes: {'ready': 30, 'backswing': 48, 'impact': 60}
            output_dir: 출력 폴더
        """
        if not os.path.exists(video_path):
            print(f"❌ 비디오 파일 없음: {video_path}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        r_idx = int(keyframes['ready'])
        b_idx = int(keyframes['backswing'])
        i_idx = int(keyframes['impact'])
        
        print(f"🚀 오버레이 생성 시작: R={r_idx}, B={b_idx}, I={i_idx}")
        
        # 1. Ready 이미지
        cap.set(cv2.CAP_PROP_POS_FRAMES, r_idx)
        ret, frame = cap.read()
        if ret:
            self.process_ready(
                frame,
                os.path.join(output_dir, "1_ready_hybrid.jpg")
            )
        
        # 2. Rotation 비디오
        self.process_rotation_video(
            cap, r_idx, i_idx, W, H,
            os.path.join(output_dir, "2_rotation_hybrid.mp4")
        )
        
        # 3. Backswing 이미지
        cap.set(cv2.CAP_PROP_POS_FRAMES, b_idx)
        ret, frame = cap.read()
        if ret:
            self.process_backswing(
                frame,
                os.path.join(output_dir, "3_backswing_hybrid.jpg")
            )
        
        # 4. Impact 이미지
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_idx)
        ret, frame = cap.read()
        if ret:
            self.process_impact(
                frame,
                os.path.join(output_dir, "4_impact_hybrid.jpg")
            )
        
        # 5. Follow 비디오
        self.process_follow_video(
            cap, i_idx, W, H,
            os.path.join(output_dir, "5_follow_hybrid.mp4")
        )
        
        cap.release()
        print(f"✅ 오버레이 생성 완료! 저장 경로: {output_dir}")
    
    def __del__(self):
        """소멸자: MediaPipe 리소스 해제"""
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()