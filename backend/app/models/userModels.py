"""
USER 모델
"""

# backend/app/models/userModels.py
from sqlalchemy import Column, String, Date
from sqlalchemy.sql import func
from app.db.base import Base

class User(Base):
    __tablename__ = "user"
    
    name = Column(String(16), nullable=False)
    id = Column(String(16), primary_key=True)
    password = Column(String(255), nullable=False)
    sex = Column(String(16))
    hand = Column(String(16))
    create_date = Column(Date, server_default=func.now())
