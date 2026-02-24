"""
User Schemas - 회원가입, 로그인, 정보 수정, 조회
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ========================================
# Request 스키마 (프론트 → 백엔드)
# ========================================

class RegisterRequest(BaseModel):
    """회원가입 요청"""
    id: str
    password: str
    name: str
    sex: Optional[str] = None   # male/female
    hand: Optional[str] = None  # right/left


class LoginRequest(BaseModel):
    """로그인 요청"""
    id: str
    password: str


class UserUpdateRequest(BaseModel):
    """사용자 정보 수정 요청"""
    name: Optional[str] = None
    password: Optional[str] = None


# ========================================
# Response 스키마 (백엔드 → 프론트)
# ========================================

class RegisterResponse(BaseModel):
    """회원가입 응답"""
    success: bool
    message: str
    user_id: str


class LoginResponse(BaseModel):
    """로그인 응답"""
    success: bool
    message: str
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str


class UserResponse(BaseModel):
    """사용자 정보 응답 (상세)"""
    id: str
    name: str
    sex: Optional[str] = None
    hand: Optional[str] = None
    create_date: datetime


class UserInfoResponse(BaseModel):
    """내 정보 조회 응답 (간단)"""
    user_id: str
    name: str
    sex: Optional[str] = None
    hand: Optional[str] = None


class UserDeleteResponse(BaseModel):
    """회원탈퇴 응답"""
    success: bool
    message: str


class CheckIdResponse(BaseModel):
    """아이디 중복 확인 응답"""
    available: bool
    message: str