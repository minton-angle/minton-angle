"""
POST 모델 (분석 세션)
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class Post(Base):
    """분석 세션"""
    __tablename__ = "post"
    
    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(36), nullable=True)  # ⭐ 일단 NULL 허용!
    type = Column(String(20), nullable=False, default='VIDEO')  # REALTIME / VIDEO
    status = Column(String(20), nullable=False, default='UPLOADED')  # UPLOADED / ANALYZING / DONE
    total_score = Column(Integer, nullable=True)
    create_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계
    files = relationship("File", back_populates="post", cascade="all, delete-orphan")
    analysis = relationship("Analysis", back_populates="post", uselist=False, cascade="all, delete-orphan")
    llm_report = relationship("LLMReport", back_populates="post", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Post {self.id} - {self.status}>"