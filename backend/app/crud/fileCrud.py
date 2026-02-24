from sqlalchemy.orm import Session
from app.models.fileModels import File

def create_file(db: Session, file_data: dict):
    """FILE 생성"""
    file = File(**file_data)
    db.add(file)
    db.commit()
    db.refresh(file)
    return file

def get_files_by_post(db: Session, post_idx: str):
    """POST의 모든 FILE 조회"""
    return db.query(File).filter(File.post_idx == post_idx).all()

def get_file_by_type(db: Session, post_idx: str, file_type: str):
    """특정 타입의 FILE 조회 (KF1, KF2, KF3)"""
    return db.query(File).filter(
        File.post_idx == post_idx,
        File.file_type == file_type
    ).first()