from sqlalchemy.orm import Session
from app.models.postModels import Post
from datetime import datetime

def create_post(db: Session, post_data: dict):
    """POST 생성"""
    post = Post(**post_data)
    db.add(post)
    db.commit()
    db.refresh(post)
    return post

def get_post(db: Session, post_idx: str):
    """POST 조회"""
    return db.query(Post).filter(Post.idx == post_idx).first()

def update_post_status(db: Session, post_idx: str, status: str, total_score: int = None):
    """POST 상태 업데이트"""
    post = db.query(Post).filter(Post.idx == post_idx).first()
    if post:
        post.status = status
        if total_score is not None:
            post.total_score = total_score
        db.commit()
        db.refresh(post)
    return post

def get_user_posts(db: Session, user_id: str, date: str = None):
    """사용자의 POST 목록 조회"""
    query = db.query(Post).filter(Post.user_id == user_id)
    
    if date:
        # 특정 날짜의 POST만 조회
        from datetime import datetime
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        query = query.filter(
            db.func.date(Post.create_date) == target_date
        )
    
    return query.order_by(Post.create_date.desc()).all()