"""
ANALYSIS 모델 (CV 분석 결과)
"""

# backend/app/models/analysisModels.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base
# from sqlalchemy.orm import relationship

class Analysis(Base):
    __tablename__ = "analysis"
    
    idx = Column(String(36), primary_key=True)  # ← 변경
    post_idx = Column(String(36), ForeignKey('post.idx', ondelete='CASCADE'), nullable=False)  # ← 변경
    kf1 = Column(Integer)
    kf2 = Column(Integer)
    kf3 = Column(Integer)
    kf1_error = Column(Float)
    kf2_error = Column(Float)
    kf3_error = Column(Float)
    score_json = Column(JSONB)
    create_date = Column(DateTime(timezone=True), server_default=func.now())