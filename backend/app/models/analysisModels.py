"""
ANALYSIS 모델 (CV 분석 결과)
"""

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class Analysis(Base):
    """자세 분석 데이터"""
    __tablename__ = "analysis"
    
    id = Column(String(36), primary_key=True)
    post_id = Column(String(36), ForeignKey('post.id', ondelete='CASCADE'), nullable=False)
    keypoint_json = Column(JSONB, nullable=False)  # 관절 좌표
    angle_json = Column(JSONB, nullable=True)  # 각도 데이터
    score_json = Column(JSONB, nullable=True)  # 부위별 점수
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계
    post = relationship("Post", back_populates="analysis")
    
    def __repr__(self):
        return f"<Analysis {self.id}>"