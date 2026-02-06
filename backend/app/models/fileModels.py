"""
FILE 모델 (업로드 파일)
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class File(Base):
    """업로드 파일"""
    __tablename__ = "file"
    
    id = Column(String(36), primary_key=True)
    post_id = Column(String(36), ForeignKey('post.id', ondelete='CASCADE'), nullable=False)
    file_type = Column(String(20), nullable=False)  # VIDEO / FRAME / THUMB
    file_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계
    post = relationship("Post", back_populates="files")
    
    def __repr__(self):
        return f"<File {self.id} - {self.file_type}>"