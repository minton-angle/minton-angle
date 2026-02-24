"""
User API Router - 회원가입, 로그인, 정보 수정, 탈퇴
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.userModels import User
from app.crud import userCrud as user_crud
from app.core.security import verify_password, create_access_token, get_current_user

from app.schemas.user import (
    RegisterRequest, RegisterResponse,
    LoginRequest, LoginResponse,
    UserResponse, UserUpdateRequest, UserDeleteResponse,
    CheckIdResponse
)

router = APIRouter(prefix="/api/auth", tags=["user"])


# ========================================
# 아이디 중복 확인
# ========================================
@router.get(
    "/check-id",
    response_model=CheckIdResponse,
    summary="아이디 중복 확인"
)
async def check_id(id: str, db: Session = Depends(get_db)):
    """아이디 중복 확인"""
    exists = user_crud.check_id_exists(db, id)
    
    if exists:
        return CheckIdResponse(
            available=False,
            message="이미 사용 중인 아이디입니다."
        )
    
    return CheckIdResponse(
        available=True,
        message="사용 가능한 아이디입니다."
    )


# ========================================
# 회원가입
# ========================================
@router.post(
    "/signup",
    response_model=RegisterResponse,
    summary="회원가입"
)
async def signup(request: RegisterRequest, db: Session = Depends(get_db)):
    """회원가입"""
    
    # 중복 확인
    if user_crud.check_id_exists(db, request.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 아이디입니다."
        )
    
    try:
        # 사용자 생성
        new_user = user_crud.create_user(
            db=db,
            user_id=request.id,
            password=request.password,
            name=request.name,
            sex=request.sex,
            hand=request.hand
        )
        
        return RegisterResponse(
            success=True,
            message="회원가입이 완료되었습니다.",
            user_id=new_user.id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"회원가입 실패: {str(e)}"
        )


# ========================================
# 로그인
# ========================================
@router.post(
    "/login",
    response_model=LoginResponse,
    summary="로그인"
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """로그인 (OAuth2 form-data)"""
    
    # 사용자 조회
    user = user_crud.get_user_by_id(db, form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다."
        )
    
    # 비밀번호 확인
    if not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다."
        )
    
    # JWT 토큰 생성
    access_token = create_access_token(data={"sub": user.id})
    
    return LoginResponse(
        success=True,
        message="로그인 성공",
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        name=user.name
    )


# ========================================
# 내 정보 조회
# ========================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="내 정보 조회"
)
async def get_my_info(
    current_user: User = Depends(get_current_user)
):
    """내 정보 조회"""
    return current_user


# ========================================
# 내 정보 수정
# ========================================
@router.put(
    "/me",
    response_model=UserResponse,
    summary="내 정보 수정 (이름, 비밀번호)"
)
async def update_my_info(
    request: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """내 정보 수정 (이름, 비밀번호)"""
    try:
        update_data = request.model_dump(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="수정할 정보를 입력해주세요."
            )
        
        # 비밀번호 길이 검증
        if 'password' in update_data:
            if len(update_data['password']) < 8:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="비밀번호는 8자 이상이어야 합니다."
                )
        
        updated_user = user_crud.update_user(
            db=db,
            user_id=current_user.id,
            name=update_data.get('name'),
            password=update_data.get('password')  # ⭐ 추가!
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다."
            )
        
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"정보 수정 실패: {str(e)}"
        )


# ========================================
# 회원탈퇴
# ========================================
@router.delete(
    "/me",
    response_model=UserDeleteResponse,
    summary="회원탈퇴"
)
async def delete_my_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """회원탈퇴"""
    try:
        success = user_crud.delete_user(db=db, user_id=current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다."
            )
        
        return UserDeleteResponse(
            success=True,
            message="회원탈퇴가 완료되었습니다. 그동안 이용해주셔서 감사합니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"회원탈퇴 실패: {str(e)}"
        )