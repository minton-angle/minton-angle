"""
POST 모델 (분석 세션)
"""

# backend/app/models/postModels.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.base import Base

class Post(Base):
    __tablename__ = "post"
    
    idx = Column(String(36), primary_key=True)  # ← 변경
    user_id = Column(String(36), ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, server_default='UPLOADED')
    total_score = Column(Integer)
    create_date = Column(DateTime(timezone=True), server_default=func.now())