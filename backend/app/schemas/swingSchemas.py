"""
swingSchemas.py
===============
스윙 분석 API 스키마
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================================
# 응답 스키마
# ============================================================

class MetricResult(BaseModel):
    """개별 지표 결과"""
    name: str
    value: float
    unit: str
    grade: str          # good, fair, poor
    score: int          # 0~100
    feedback: str


class PhaseResult(BaseModel):
    """구간별 결과"""
    name: str
    display_type: str   # "image" or "video"
    file_url: str
    impact_image_url: Optional[str] = None
    metrics: Dict[str, MetricResult]
    feedback: str


class OverallResult(BaseModel):
    """종합 결과"""
    score: int
    grade: str          # excellent, good, fair, poor
    feedback_summary: List[str]


class VideoInfo(BaseModel):
    """영상 정보"""
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float


class SwingAnalyzeResponse(BaseModel):
    """분석 응답"""
    success: bool
    analysis_id: str
    video_info: VideoInfo
    overall: OverallResult
    phases: Dict[str, PhaseResult]
    raw_metrics: Dict[str, float]
    created_at: str


class SwingAnalyzeError(BaseModel):
    """에러 응답"""
    success: bool = False
    message: str


# ============================================================
# GT 관련 스키마
# ============================================================

class ExpertMetrics(BaseModel):
    """전문가 1명 지표"""
    name: str
    video_path: str
    total_frames: int
    impact_frame: int
    elbow_angle: float
    impact_height: float
    hip_rotation: float
    followthrough: float


class CriteriaRange(BaseModel):
    """판정 기준 범위"""
    expert_min: float
    expert_max: float
    expert_mean: float
    expert_std: float
    good: tuple
    fair: tuple


class GTCreateResponse(BaseModel):
    """GT 생성 응답"""
    success: bool = True
    created_at: str
    num_experts: int
    experts: List[ExpertMetrics]
    criteria: Dict[str, CriteriaRange]


class CriteriaResponse(BaseModel):
    """판정 기준 조회 응답"""
    criteria: Dict[str, Any]


# ============================================================
# 전문가 영상 관련
# ============================================================

class ExpertVideoResponse(BaseModel):
    """전문가 영상 정보"""
    url: str
    type: str = "local"  # local or youtube
    available: bool = True
