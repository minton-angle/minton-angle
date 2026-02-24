import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional
import base64

class PoseDetector:
    """MediaPipe 기반 Pose Detection"""
    
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
            # 🆕 디버깅 로그 1: 입력 확인
            print(f"  📥 입력 길이: {len(base64_image)}, 시작: {base64_image[:50]}")
            
            # Base64 디코딩
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            # 🆕 디버깅 로그 2: 바이트 변환
            image_bytes = base64.b64decode(base64_image)
            print(f"  ✅ 바이트 변환: {len(image_bytes)} bytes")
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"  ❌ cv2.imdecode 실패!")
                return None
            
            # 🆕 디버깅 로그 3: 이미지 크기
            print(f"  ✅ 이미지 디코딩: {image.shape}")
            
            # RGB 변환 (MediaPipe는 RGB 필요)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Pose Detection
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"  ⚠️ pose_landmarks 없음 (사람 미감지)")
                return None
            
            # 🆕 디버깅 로그 4: 성공
            print(f"  ✅ Keypoint 추출 성공!")
            
            # 19개 keypoint 추출
            keypoints = {}
            for idx, name in self.selected_keypoints.items():
                landmark = results.pose_landmarks.landmark[idx]
                keypoints[f'{name}_x'] = landmark.x
                keypoints[f'{name}_y'] = landmark.y
                keypoints[f'{name}_z'] = landmark.z
                keypoints[f'{name}_visibility'] = landmark.visibility
            
            return keypoints
            
        except Exception as e:
            print(f"  ❌ Keypoint extraction error: {e}")
            import traceback
            traceback.print_exc()
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
    
    def __del__(self):
        """소멸자: MediaPipe 리소스 해제"""
        if hasattr(self, 'pose'):
            self.pose.close()