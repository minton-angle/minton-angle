"""
POST 모델 (분석 세션)
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.base import Base

class Post(Base):
    __tablename__ = "post"
    
    idx = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(20), nullable=False)  # 'REALTIME' | 'VIDEO'
    swing_num = Column(Integer, nullable=True)  # 1, 2, 3 (REALTIME일 때만)
    status = Column(String(20), nullable=False, default='UPLOADED')
    total_score = Column(Integer, nullable=True)
    create_date = Column(DateTime, default=func.now())