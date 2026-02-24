"""
보안 관련 유틸리티
- 비밀번호 해싱
- JWT 토큰 생성/검증
- 현재 사용자 인증
"""
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

# ========================================
# 설정값
# ========================================
SECRET_KEY = "minton-angle-secret-key-2026"  # 나중에 환경변수로!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7  # 7일

# ========================================
# OAuth2 스키마
# ========================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ========================================
# 비밀번호 해싱
# ========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    return pwd_context.verify(plain_password, hashed_password)

# ========================================
# JWT 토큰
# ========================================
def create_access_token(data: dict) -> str:
    """JWT 토큰 생성"""
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[str]:
    """
    JWT 토큰 디코딩
    
    Returns:
        user_id (성공)
        None (실패/만료)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")  # ⭐ "sub"로 통일!
        return user_id
    
    except JWTError:
        return None

# ========================================
# 현재 사용자 조회
# ========================================
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(lambda: next(__import__('app.db.session', fromlist=['get_db']).get_db()))
):
    """
    JWT 토큰으로 현재 사용자 조회
    
    Args:
        token: Bearer 토큰
        db: 데이터베이스 세션
    
    Returns:
        User: 인증된 사용자 객체
    
    Raises:
        HTTPException: 인증 실패 시
    """
    from app.crud import userCrud
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보가 유효하지 않습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")  # ⭐ "sub"로 통일!
        
        if user_id is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = userCrud.get_user_by_id(db, user_id)
    
    if user is None:
        raise credentials_exception
    
    return user