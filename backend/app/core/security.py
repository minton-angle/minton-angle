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

# ⭐ 개발 모드 (배포 시 False로 변경!)
DEV_MODE = True

# ========================================
# OAuth2 스키마
# ========================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

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
        user_id: str = payload.get("sub")
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
    - DEV_MODE=False 상태에서 실제 토큰 검증 수행
    """
    from app.models.userModels import User
    from app.crud import userCrud

    # 인증 실패 시 던질 공통 에러 설정
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보가 유효하지 않거나 만료되었습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. 토큰 존재 여부 확인
    if not token:
        print("❌ 인증 실패: 토큰이 없습니다.")
        raise credentials_exception

    try:
        # 2. JWT 토큰 디코딩 및 만료 체크
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            print("❌ 인증 실패: 토큰에 유저 ID(sub) 정보가 없습니다.")
            raise credentials_exception
            
    except JWTError as e:
        print(f"❌ 인증 실패: 토큰 해석 중 오류 발생 ({str(e)})")
        raise credentials_exception

    # 3. DB에서 실제 유저 조회
    user = userCrud.get_user_by_id(db, user_id)
    
    if user is None:
        print(f"❌ 인증 실패: DB에 유저({user_id})가 존재하지 않습니다.")
        raise credentials_exception

    # 4. 인증 성공 (실제 유저 객체 반환)
    print(f"✅ 인증 성공: {user.id} 님 환영합니다.")
    return user