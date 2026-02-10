from sqlalchemy.orm import Session
from app.models.analysisModels import Analysis

def create_analysis(db: Session, analysis_data: dict):
    """ANALYSIS 생성"""
    analysis = Analysis(**analysis_data)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def get_analysis_by_post(db: Session, post_idx: str):
    """POST의 ANALYSIS 조회"""
    return db.query(Analysis).filter(Analysis.post_idx == post_idx).first()

def update_analysis_errors(db: Session, analysis_idx: str, kf1_error: float, kf2_error: float, kf3_error: float):
    """오차값 업데이트"""
    analysis = db.query(Analysis).filter(Analysis.idx == analysis_idx).first()
    if analysis:
        analysis.kf1_error = kf1_error
        analysis.kf2_error = kf2_error
        analysis.kf3_error = kf3_error
        db.commit()
        db.refresh(analysis)
    return analysis