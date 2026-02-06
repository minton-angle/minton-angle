"""
LLM_REPORT 모델 (LLM 피드백)
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class LLMReport(Base):
    """LLM 분석 리포트"""
    __tablename__ = "llm_report"
    
    id = Column(String(36), primary_key=True)
    post_id = Column(String(36), ForeignKey('post.id', ondelete='CASCADE'), nullable=False)
    summary = Column(Text, nullable=False)  # 한 줄 총평
    key_points = Column(JSON, nullable=True)  # Key Point 1~3
    improvement = Column(JSON, nullable=True)  # 개선 제안
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계
    post = relationship("Post", back_populates="llm_report")
    
    def __repr__(self):
        return f"<LLMReport {self.id}>"