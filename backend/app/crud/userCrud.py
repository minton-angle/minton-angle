"""
User CRUD 작업
"""
from sqlalchemy.orm import Session
from app.models.userModels import User
from app.core.security import hash_password
from typing import Optional


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """ID로 사용자 조회"""
    return db.query(User).filter(User.id == user_id).first()


def check_id_exists(db: Session, user_id: str) -> bool:
    """아이디 중복 확인"""
    return db.query(User).filter(User.id == user_id).first() is not None


def create_user(
    db: Session,
    user_id: str,
    password: str,
    name: str,
    sex: Optional[str] = None,
    hand: Optional[str] = None
) -> User:
    """사용자 생성"""
    hashed_pw = hash_password(password)
    
    new_user = User(
        id=user_id,
        password=hashed_pw,
        name=name,
        sex=sex,
        hand=hand
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


def update_user(
    db: Session,
    user_id: str,
    name: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[User]:
    """사용자 정보 수정 (이름 + 비밀번호)"""
    from app.core.security import hash_password
    
    user = get_user_by_id(db, user_id)
    
    if not user:
        return None
    
    if name is not None:
        user.name = name
    
    if password is not None:
        user.password = hash_password(password)
    
    db.commit()
    db.refresh(user)
    
    return user


def delete_user(db: Session, user_id: str) -> bool:
    """사용자 삭제 (회원탈퇴)"""
    user = db.query(User).filter(User.id == user_id).first() 
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True