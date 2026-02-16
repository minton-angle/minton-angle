"""
Swing Analysis Schemas
스윙 분석 관련 Request/Response 스키마 = DTO
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class PostTypeEnum(str, Enum):
    """POST 타입"""
    REALTIME = "REALTIME"
    VIDEO = "VIDEO"


class FileTypeEnum(str, Enum):
    """파일 타입"""
    KF1 = "KF1"
    KF2 = "KF2"
    KF3 = "KF3"
    VIDEO = "VIDEO"


# ==================== Request Schemas ====================

class SwingAnalysisRequest(BaseModel):
    """실시간 스윙 분석 요청"""
    user_id: str = Field(..., description="사용자 ID")
    swing_num: int = Field(..., ge=1, le=3, description="스윙 횟수 (1~3)")
    post_id: Optional[str] = Field(None, description="POST ID (2~3회차 시 필수, 1회차에서 받은 ID)")
    keypoints: List[Dict[str, float]] = Field(..., description="키포인트 데이터 [19개 랜드마크 x 3개 값(x,y,z)]")
    frames: Optional[List[str]] = Field(None, description="Base64 인코딩된 프레임 이미지들 (키프레임 저장용)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "swing_num": 2,
                "post_id": "550e8400-e29b-41d4-a716-446655440000",  # 2~3회차는 필수
                "keypoints": [
                    {
                        "nose_x": 0.5,
                        "nose_y": 0.3,
                        "nose_z": 0.1,
                        "left_shoulder_x": 0.4,
                        "left_shoulder_y": 0.2,
                        "left_shoulder_z": 0.05,
                        # ... (57개 키)
                    }
                ],
                "frames": ["data:image/jpeg;base64,/9j/4AAQ...", "..."]
            }
        }


class VideoUploadRequest(BaseModel):
    """동영상 업로드 분석 요청"""
    user_id: str = Field(..., description="사용자 ID")
    video_file: str = Field(..., description="Base64 인코딩된 동영상 파일")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "video_file": "data:video/mp4;base64,AAAAIGZ0eXBpc29t..."
            }
        }


# ==================== Response Schemas ====================

class ScoreDetail(BaseModel):
    """6대 지표 점수 상세"""
    elbow_height: int = Field(..., ge=0, le=100, description="팔꿈치 높이")
    wrist_snap: int = Field(..., ge=0, le=100, description="손목 스냅")
    hit_position: int = Field(..., ge=0, le=100, description="타구 위치")
    shoulder_rotation: int = Field(..., ge=0, le=100, description="어깨 회전")
    racket_angle: int = Field(..., ge=0, le=100, description="라켓 각도")
    follow_through: int = Field(..., ge=0, le=100, description="팔로우스루")
    
    class Config:
        json_schema_extra = {
            "example": {
                "elbow_height": 85,
                "wrist_snap": 78,
                "hit_position": 90,
                "shoulder_rotation": 82,
                "racket_angle": 88,
                "follow_through": 75
            }
        }


class QuickFeedbackResponse(BaseModel):
    """빠른 피드백 응답 (1~2회차)"""
    swing_num: int = Field(..., description="현재 스윙 횟수")
    post_id: str = Field(..., description="POST ID (다음 회차에서 사용)")
    quick_feedback: str = Field(..., description="간단한 피드백 메시지")
    save_to_db: bool = Field(True, description="DB 저장 여부")
    scores: Optional[ScoreDetail] = Field(None, description="점수 상세 (참고용)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "swing_num": 1,
                "post_id": "550e8400-e29b-41d4-a716-446655440000",
                "quick_feedback": "잘하고 있어요! 😊",
                "save_to_db": True,
                "scores": {
                    "elbow_height": 85,
                    "wrist_snap": 78,
                    "hit_position": 90,
                    "shoulder_rotation": 82,
                    "racket_angle": 88,
                    "follow_through": 75
                }
            }
        }


class AnalysisCompleteResponse(BaseModel):
    """분석 완료 응답 (3회차 - DB 저장)"""
    swing_num: int = Field(..., description="현재 스윙 횟수")
    post_id: str = Field(..., description="생성된 POST ID")
    save_to_db: bool = Field(True, description="DB 저장 여부")
    total_score: int = Field(..., ge=0, le=100, description="종합 점수")
    scores: ScoreDetail = Field(..., description="6대 지표 점수")
    quick_feedback: str = Field(..., description="간단한 피드백 메시지")
    
    class Config:
        json_schema_extra = {
            "example": {
                "swing_num": 3,
                "post_id": "550e8400-e29b-41d4-a716-446655440000",
                "save_to_db": True,
                "total_score": 87,
                "scores": {
                    "elbow_height": 85,
                    "wrist_snap": 78,
                    "hit_position": 90,
                    "shoulder_rotation": 82,
                    "racket_angle": 88,
                    "follow_through": 75
                },
                "quick_feedback": "아주 좋아요! 👍"
            }
        }


# ==================== Internal Use (DB -> Response 변환용) ====================

class KeyframeInfo(BaseModel):
    """키프레임 정보"""
    kf1: int = Field(..., description="준비자세 프레임 인덱스")
    kf2: int = Field(..., description="백스윙 프레임 인덱스")
    kf3: int = Field(..., description="임팩트 프레임 인덱스")


class KeyframeImages(BaseModel):
    """키프레임 이미지 경로"""
    kf1: str = Field(..., description="준비자세 이미지 경로")
    kf2: str = Field(..., description="백스윙 이미지 경로")
    kf3: str = Field(..., description="임팩트 이미지 경로")