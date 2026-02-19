"""
회원가입/로그인 Request/Response 스키마
"""
from pydantic import BaseModel
from typing import Optional

# ========================================
# Request 스키마 (프론트 → 백엔드)
# ========================================

class RegisterRequest(BaseModel):
    """회원가입 요청"""
    id: str           # 로그인 ID
    password: str     # 비밀번호
    name: str         # 닉네임
    sex: Optional[str] = None   # male/female
    hand: Optional[str] = None  # right/left


class LoginRequest(BaseModel):
    """로그인 요청"""
    id: str
    password: str


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
    access_token: str    # JWT 토큰
    token_type: str      # "bearer"
    user_id: str
    name: str


class UserInfoResponse(BaseModel):
    """내 정보 조회 응답"""
    user_id: str
    name: str
    sex: Optional[str]
    hand: Optional[str]