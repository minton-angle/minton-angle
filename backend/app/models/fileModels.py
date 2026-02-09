"""
FILE 모델 (업로드 파일)
"""

# backend/app/models/fileModels.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.base import Base

class File(Base):
    __tablename__ = "file"
    
    idx = Column(String(36), primary_key=True)  # ← 변경
    post_idx = Column(String(36), ForeignKey('post.idx', ondelete='CASCADE'), nullable=False)  # ← 변경
    file_type = Column(String(20), nullable=False)
    file_name = Column(String(255))
    file_extension = Column(String(10))  # ← 추가
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    storage_type = Column(String(10))
    s3_bucket = Column(String(100))  # ← 추가
    s3_key = Column(String(500))  # ← 추가
    create_date = Column(DateTime(timezone=True), server_default=func.now())