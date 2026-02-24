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
    keypoints: List[List[float]] = Field(..., description="키포인트 데이터 [33개 랜드마크 x 4개 값(x,y,z,visibility)]")
    frames: Optional[List[str]] = Field(None, description="Base64 인코딩된 프레임 이미지들 (키프레임 저장용)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "swing_num": 3,
                "post_id": "2bf25d4c-d698-4a37-b56e-5de29f3c800a",  # 2~3회차는 필수
                "keypoints": [
                    [0.5, 0.3, 0.8, 0.9],
                    [0.4, 0.5, 0.7, 0.95],
                    # ... 33개
                ]
                # "frames": ["data:image/jpeg;base64,/9j/4AAQ...", "..."]
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


class AnalysisCompleteResponse(BaseModel): # 프론트 측에서 기대하는 최종 응답 형태
    """분석 완료 응답 (3회차 - DB 저장)"""
    swing_num: int = Field(..., description="현재 스윙 횟수")
    post_id: str = Field(..., description="생성된 POST ID")
    save_to_db: bool = Field(True, description="DB 저장 여부")
    total_score: int = Field(..., ge=0, le=100, description="종합 점수")
    scores: ScoreDetail = Field(..., description="6대 지표 점수")
    quick_feedback: str = Field(..., description="간단한 피드백 메시지")
    # 추가
    detailed_feedback: Optional[Dict[str, Any]] = None
    llm_report_idx: Optional[str] = None  # LLM 보고서와 연결할 수 있는 ID (선택적)
    
    class Config:
        json_schema_extra = {
            "example": {
                "swing_num": 3,
                "post_id": "550e8400-e29b-41d4-a716-446655440000", # 1회차에서 생성된 POST(세션/분석 묶음)을 가리키는 ID (3회차에도 동일하게 사용)
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
                "quick_feedback": "아주 좋아요! 👍",
                "detailed_feedback": {
                    "elbow_height": "팔꿈치가 너무 낮아요. 좀 더 높게 유지해보세요.",
                    "wrist_snap": "손목 스냅이 부족해요. 임팩트 순간에 손목을 더 빠르게 휘둘러보세요."
                },
                "llm_report_idx": "2d9285fd-fb6e-4c49-bfb2-edf5cec4bca5"  #  “생성된 LLM 리포트 한 건”을 가리키는 ID

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