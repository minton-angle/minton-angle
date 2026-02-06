"""
USER 모델
"""

from sqlalchemy import Column, String, Text, Date
from sqlalchemy.sql import func

from app.db.base import Base


class User(Base):
    """사용자"""
    __tablename__ = "user"
    
    id = Column(String(16), primary_key=True)  # 사용자 ID (PK)
    name = Column(String(16), nullable=False)  # 사용자 이름
    password = Column(Text, nullable=False)  # 비밀번호
    create_date = Column(Date, server_default=func.now())  # 가입일
    
    def __repr__(self):
        return f"<User {self.id} - {self.name}>"
