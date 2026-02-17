"""
보안 관련 유틸리티
- 비밀번호 해싱
- JWT 토큰 생성/검증
"""
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

# ========================================
# 설정값
# ========================================
SECRET_KEY = "minton-angle-secret-key-2026"  # 나중에 환경변수로!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7  # 7일

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
def create_access_token(user_id: str) -> str:
    """JWT 토큰 생성"""
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "user_id": user_id,
        "exp": expire
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[str]:
    """
    JWT 토큰 디코딩
    
    Returns:
        user_id (성공)
        None (실패/만료)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        return user_id
    
    except JWTError:
        return None