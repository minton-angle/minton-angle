"""
Pose Detection: MediaPipe 기반 실시간 keypoint 추출
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional
import base64


class PoseDetector:
    """MediaPipe 기반 Pose Detection (실시간 추출)"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe 33개 keypoint 중 19개 선정
        self.selected_keypoints = {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            17: 'left_pinky',
            18: 'right_pinky',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle',
            29: 'left_heel',
            30: 'right_heel',
            31: 'left_foot_index',
            32: 'right_foot_index'
        }
        
        # MediaPipe 전체 매핑 (overlay용)
        self.mp_pose_landmark = self.mp_pose.PoseLandmark
    
    def extract_from_base64(self, base64_image: str) -> Optional[Dict]:
        """
        Base64 이미지에서 keypoint 추출
        
        Args:
            base64_image: "data:image/jpeg;base64,..." 형식
            
        Returns:
            {
                'nose_x': 0.512,
                'nose_y': 0.234,
                'nose_z': 0.012,
                'nose_visibility': 0.98,
                ...
            }
        """
        try:
            # Base64 디코딩
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"  ❌ cv2.imdecode 실패!")
                return None
            
            # RGB 변환 (MediaPipe는 RGB 필요)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Pose Detection
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"  ⚠️ pose_landmarks 없음 (사람 미감지)")
                return None
            
            # 19개 keypoint 추출
            keypoints = {}
            for idx, name in self.selected_keypoints.items():
                landmark = results.pose_landmarks.landmark[idx]
                keypoints[f'{name}_x'] = landmark.x
                keypoints[f'{name}_y'] = landmark.y
                keypoints[f'{name}_z'] = landmark.z
                # keypoints[f'{name}_visibility'] = landmark.visibility
            
            return keypoints
            
        except Exception as e:
            print(f"  ❌ Keypoint extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        OpenCV 프레임에서 keypoint 추출 (실시간 처리용)
        
        Args:
            frame: OpenCV 이미지 (BGR)
            
        Returns:
            {
                'nose': (x_px, y_px),
                'right_shoulder': (x_px, y_px),
                ...
            }
            또는
            {
                'nose_x': 0.512, 'nose_y': 0.234, ...
            } (normalized=True일 때)
        """
        try:
            h, w, _ = frame.shape
            
            # RGB 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            kps = {}
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                for idx, name in self.selected_keypoints.items():
                    lm = landmarks[idx]
                    
                    # visibility 체크
                    if lm.visibility > 0.5:
                        # 픽셀 좌표로 변환
                        kps[name] = (int(lm.x * w), int(lm.y * h))
            
            return kps
            
        except Exception as e:
            print(f"  ❌ Frame extraction error: {e}")
            return None
    
    def extract_from_frame_normalized(self, frame: np.ndarray) -> Optional[Dict]:
        """
        OpenCV 프레임에서 정규화된 keypoint 추출
        
        Args:
            frame: OpenCV 이미지 (BGR)
            
        Returns:
            {
                'nose_x': 0.512,
                'nose_y': 0.234,
                'nose_z': 0.012,
                'nose_visibility': 0.98,
                ...
            }
        """
        try:
            # RGB 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if not results.pose_landmarks:
                return None
            
            # 19개 keypoint 추출 (정규화 좌표)
            keypoints = {}
            for idx, name in self.selected_keypoints.items():
                landmark = results.pose_landmarks.landmark[idx]
                keypoints[f'{name}_x'] = landmark.x
                keypoints[f'{name}_y'] = landmark.y
                keypoints[f'{name}_z'] = landmark.z
                keypoints[f'{name}_visibility'] = landmark.visibility
            
            return keypoints
            
        except Exception as e:
            print(f"  ❌ Normalized extraction error: {e}")
            return None
    
    def extract_from_frames(self, frames: List[str]) -> List[Dict]:
        """
        여러 Base64 프레임에서 keypoint 추출
        
        Args:
            frames: [base64_image1, base64_image2, ...]
            
        Returns:
            [
                {'frame_id': 0, 'nose_x': 0.512, ...},
                {'frame_id': 1, 'nose_x': 0.513, ...},
                ...
            ]
        """
        keypoints_list = []
        
        for frame_id, frame in enumerate(frames):
            keypoints = self.extract_from_base64(frame)
            
            if keypoints:
                keypoints['frame_id'] = frame_id
                keypoints_list.append(keypoints)
        
        return keypoints_list
    
    def extract_from_video(self, video_path: str) -> List[Dict]:
        """
        비디오 파일에서 모든 프레임의 keypoint 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            [
                {'frame_id': 0, 'timestamp': 0.0, 'nose_x': 0.512, ...},
                ...
            ]
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        
        keypoints_list = []
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            keypoints = self.extract_from_frame_normalized(frame)
            
            if keypoints:
                keypoints['frame_id'] = frame_id
                keypoints['timestamp'] = frame_id / fps
                keypoints_list.append(keypoints)
            
            frame_id += 1
            
            if frame_id % 10 == 0:
                print(f"      → {frame_id}프레임...", end='\r')
        
        cap.release()
        print(f"   ✅ 완료: {frame_id}프레임                    ")
        
        return keypoints_list
    
    def __del__(self):
        """소멸자: MediaPipe 리소스 해제"""
        if hasattr(self, 'pose'):
            self.pose.close()