"""
LLM_REPORT 모델 (LLM 피드백)
"""

# backend/app/models/llmReportModels.py
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import func
from app.db.base import Base

class LLMReport(Base):
    __tablename__ = "llm_report"
    
    idx = Column(String(36), primary_key=True)  # ← 변경
    post_idx = Column(String(36), ForeignKey('post.idx', ondelete='CASCADE'), nullable=False)  # ← 변경
    feedback = Column(JSON, nullable=False)  # ← summary 삭제
    create_date = Column(DateTime(timezone=True), server_default=func.now())