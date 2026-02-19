"""
회원가입/로그인 API
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional

from app.db.session import get_db
from app.models.userModels import User
from app.schemas.auth import (
    RegisterRequest, RegisterResponse,
    LoginResponse,
    UserInfoResponse
)
from app.core.security import (
    verify_password,
    create_access_token, decode_access_token
)
from app.crud import userCrud

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ========================================
# FastAPI Security 클래스
# ========================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ========================================
# 공통 함수: 현재 유저 가져오기
# ========================================
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    토큰 검증 후 현재 유저 반환
    - Depends()로 다른 API에서 재사용 가능!
    """
    # 토큰 디코딩
    user_id = decode_access_token(token)
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="토큰이 유효하지 않습니다."
        )
    
    # ✅ CRUD 사용!
    user = userCrud.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다."
        )
    
    return user


# ========================================
# API 엔드포인트
# ========================================

@router.get("/check-id")
async def check_id_duplicate(
    id: str = Query(..., description="확인할 아이디"),
    db: Session = Depends(get_db)
):
    """아이디 중복 확인"""
    
    is_duplicate = userCrud.check_id_exists(db, id)
    
    if is_duplicate:
        return {
            "available": False,
            "message": "이미 사용중인 아이디입니다."
        }
    
    return {
        "available": True,
        "message": "사용 가능한 아이디입니다."
    }


@router.post("/signup", response_model=RegisterResponse)
async def signup(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):
    """회원가입"""
    
    # ✅ CRUD 사용!
    if userCrud.check_id_exists(db, request.id):
        raise HTTPException(
            status_code=400,
            detail="이미 사용중인 아이디입니다."
        )
    
    # ✅ CRUD 사용!
    new_user = userCrud.create_user(
        db=db,
        user_id=request.id,
        password=request.password,
        name=request.name,
        sex=request.sex,
        hand=request.hand
    )
    
    return RegisterResponse(
        success=True,
        message="회원가입이 완료되었습니다!",
        user_id=new_user.id
    )


# ========================================
# 로그인 (form-data 방식 - OAuth2 정석!)
# ========================================
@router.post("/login", response_model=LoginResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """로그인 (OAuth2 표준 방식)"""
    
    # ✅ CRUD 사용!
    user = userCrud.get_user_by_id(db, form.username)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 틀렸습니다."
        )
    
    if not verify_password(form.password, user.password):
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 틀렸습니다."
        )
    
    access_token = create_access_token(user.id)
    
    return LoginResponse(
        success=True,
        message=f"{user.name}님 환영합니다!",
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        name=user.name
    )


@router.get("/me", response_model=UserInfoResponse)
async def get_my_info(
    current_user = Depends(get_current_user)
):
    """내 정보 조회 (토큰 필요)"""
    
    return UserInfoResponse(
        user_id=current_user.id,
        name=current_user.name,
        sex=current_user.sex,
        hand=current_user.hand
    )


