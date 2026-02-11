"""
pose_extractor.py
=================
MediaPipe를 사용한 포즈 추출

영상에서 프레임별 관절 좌표를 추출하고 정규화
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# MediaPipe import (버전에 따라 다름)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW_API = True
except ImportError:
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        MEDIAPIPE_NEW_API = False
    except:
        MEDIAPIPE_NEW_API = None
        print("⚠️ MediaPipe 설치 필요: pip install mediapipe")


@dataclass
class FrameLandmarks:
    """한 프레임의 관절 좌표"""
    frame_idx: int
    landmarks: Dict[str, Tuple[float, float, float]]  # {joint_name: (x, y, visibility)}
    timestamp_ms: float


class PoseExtractor:
    """
    영상에서 포즈 추출
    
    사용법:
        extractor = PoseExtractor()
        frames_data = extractor.extract_from_video("video.mp4")
    """
    
    # MediaPipe 관절 인덱스
    LANDMARK_INDICES = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: MediaPipe 모델 경로 (새 API용)
        """
        self.pose = None
        self._init_mediapipe(model_path)
    
    def _init_mediapipe(self, model_path: Optional[str] = None):
        """MediaPipe 초기화 (버전에 따라 다르게)"""
        if MEDIAPIPE_NEW_API is True:
            # 새 API (0.10.x)
            if model_path and Path(model_path).exists():
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO
                )
                self.pose = vision.PoseLandmarker.create_from_options(options)
            else:
                print("⚠️ 새 MediaPipe API는 모델 파일이 필요합니다.")
                self.pose = None
                
        elif MEDIAPIPE_NEW_API is False:
            # 구 API (0.9.x 이하)
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.pose = None
    
    def extract_from_video(self, video_path: str) -> List[FrameLandmarks]:
        """
        영상에서 모든 프레임의 관절 좌표 추출
        
        Args:
            video_path: 영상 파일 경로
            
        Returns:
            List[FrameLandmarks]: 프레임별 관절 데이터
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self._process_frame(rgb, frame_idx, fps)
            
            if landmarks:
                frames_data.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        if self.pose and hasattr(self.pose, 'close'):
            self.pose.close()
        
        return frames_data
    
    def _process_frame(self, rgb_frame: np.ndarray, frame_idx: int, fps: float) -> Optional[FrameLandmarks]:
        """단일 프레임 처리"""
        if self.pose is None:
            return self._dummy_landmarks(frame_idx, fps)
        
        if MEDIAPIPE_NEW_API:
            return self._process_new_api(rgb_frame, frame_idx, fps)
        else:
            return self._process_old_api(rgb_frame, frame_idx, fps)
    
    def _process_old_api(self, rgb_frame: np.ndarray, frame_idx: int, fps: float) -> Optional[FrameLandmarks]:
        """구 API로 처리"""
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        for name, idx in self.LANDMARK_INDICES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = (lm.x, lm.y, lm.visibility)
        
        return FrameLandmarks(
            frame_idx=frame_idx,
            landmarks=landmarks,
            timestamp_ms=frame_idx / fps * 1000
        )
    
    def _process_new_api(self, rgb_frame: np.ndarray, frame_idx: int, fps: float) -> Optional[FrameLandmarks]:
        """새 API로 처리"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx / fps * 1000)
        
        results = self.pose.detect_for_video(mp_image, timestamp_ms)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        pose_landmarks = results.pose_landmarks[0]
        
        landmarks = {}
        for name, idx in self.LANDMARK_INDICES.items():
            lm = pose_landmarks[idx]
            landmarks[name] = (lm.x, lm.y, lm.visibility)
        
        return FrameLandmarks(
            frame_idx=frame_idx,
            landmarks=landmarks,
            timestamp_ms=timestamp_ms
        )
    
    def _dummy_landmarks(self, frame_idx: int, fps: float) -> FrameLandmarks:
        """MediaPipe 없을 때 더미 데이터 (테스트용)"""
        landmarks = {
            name: (0.5, 0.5, 1.0) for name in self.LANDMARK_INDICES
        }
        return FrameLandmarks(
            frame_idx=frame_idx,
            landmarks=landmarks,
            timestamp_ms=frame_idx / fps * 1000
        )
    
    def normalize_landmarks(self, frames_data: List[FrameLandmarks]) -> List[FrameLandmarks]:
        """
        관절 좌표 정규화 (어깨 기준)
        
        - 오른쪽 어깨를 원점으로
        - 상체 길이로 스케일 정규화
        """
        normalized = []
        
        for frame in frames_data:
            lm = frame.landmarks
            
            # 기준점: 오른쪽 어깨
            origin_x = lm["right_shoulder"][0]
            origin_y = lm["right_shoulder"][1]
            
            # 스케일: 어깨-골반 거리
            scale = abs(lm["right_shoulder"][1] - lm["right_hip"][1])
            if scale < 0.01:
                scale = 0.3  # 기본값
            
            # 정규화
            norm_landmarks = {}
            for name, (x, y, v) in lm.items():
                norm_x = (x - origin_x) / scale
                norm_y = (y - origin_y) / scale
                norm_landmarks[name] = (norm_x, norm_y, v)
            
            normalized.append(FrameLandmarks(
                frame_idx=frame.frame_idx,
                landmarks=norm_landmarks,
                timestamp_ms=frame.timestamp_ms
            ))
        
        return normalized


def get_video_info(video_path: str) -> Dict:
    """영상 정보 조회"""
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frame_image(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """특정 프레임 이미지 추출"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_video_clip(video_path: str, start_frame: int, end_frame: int, output_path: str) -> bool:
    """영상 구간 추출"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    return Path(output_path).exists()
