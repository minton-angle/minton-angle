from typing import Any, Dict, Optional, List
import logging
import time
import json
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.report.LLM_Total_report import generate_report
from app.db.session import SessionLocal
from app.models.postModels import Post
from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ 프론트엔드 호출과 맞추기 위해 prefix를 유지하거나 필요시 조정하세요.
router = APIRouter(prefix="/api/report", tags=["TotalReport"])
logger_api = logging.getLogger("app.api")

# --- 통계 및 계산 유틸리티 (기존 로직 유지) ---
def _mean_abs_kf_error(a: Analysis) -> float:
    vals = [abs(float(v)) for v in (a.kf1_error, a.kf2_error, a.kf3_error) if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

def _series_vals(analyses: List[Analysis], key: str) -> List[float]:
    out = []
    for a in analyses:
        v = getattr(a, key, None)
        if v is not None:
            try: out.append(abs(float(v)))
            except: continue
    return out

def _std(arr: List[float]) -> float:
    if not arr: return 0.0
    m = sum(arr) / len(arr)
    return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5

def _compute_growth_insights(analyses: List[Analysis]) -> Dict[str, Any]:
    if not analyses: return {"growth": None, "consistency": None}
    keys = ["kf1_error", "kf2_error", "kf3_error"]
    stds = {k: _std(_series_vals(analyses, k)) for k in keys}
    volatile = max(stds.items(), key=lambda kv: kv[1])[0] if stds else None
    first_mean = _mean_abs_kf_error(analyses[0])
    last_mean = _mean_abs_kf_error(analyses[-1])
    delta = last_mean - first_mean
    return {
        "growth": {
            "direction": "improved" if delta < -1e-9 else ("worsened" if delta > 1e-9 else "flat"),
            "delta_mean_abs_kf_error": round(delta, 4),
            "first_mean_abs_kf_error": round(first_mean, 4),
            "last_mean_abs_kf_error": round(last_mean, 4),
        },
        "consistency": {"kf": volatile, "std": round(stds.get(volatile, 0.0), 4)} if volatile else None,
    }

# ------------------------------------------------------------------
# ⭐ [수정] 사용자 통합 히스토리 조회 API
# ------------------------------------------------------------------
@router.get("/analysis/user/{user_id}")
def get_user_total_history(user_id: str, range: str = "7d", db: Session = Depends(get_db)):
    """user_id에 해당하는 모든 Post를 찾아 통합 분석 데이터를 반환합니다."""
    try:
        r = (range or "7d").lower().strip()
        now = datetime.utcnow()
        delta_map = {"7d": 7, "1m": 30, "3m": 90}
        
        # 1. 해당 사용자의 모든 Post ID 리스트 가져오기
        user_posts = db.query(Post.idx).filter(Post.user_id == user_id).all()
        post_indices = [p.idx for p in user_posts]
        
        if not post_indices:
            raise HTTPException(status_code=404, detail="해당 사용자의 분석 데이터가 없습니다.")

        # 2. 해당 Post들에 연결된 모든 Analysis 조회
        query = db.query(Analysis).filter(Analysis.post_idx.in_(post_indices))
        if r in delta_map:
            query = query.filter(Analysis.create_date >= now - timedelta(days=delta_map[r]))
        
        analyses = query.order_by(Analysis.create_date.asc()).all()
        
        if not analyses:
            raise HTTPException(status_code=404, detail="해당 기간 내 분석 데이터가 없습니다.")

        sessions = []
        for a in analyses:
            sj = a.score_json or {}
            # 프론트 차트 렌더링용 구조 변환
            sessions.append({
                "idx": a.idx,
                "created_at": a.create_date.isoformat() if a.create_date else None,
                "score": sj.get("total_score", 0),
                "kf_error": round(_mean_abs_kf_error(a), 4),
                "stage_scores": {
                    "1_Ready_Total": sj.get("details", {}).get("Ready", {}).get("Ready_score", 0),
                    "2_Rotation_Total": sj.get("details", {}).get("Rotation", {}).get("Rotation_score", 0),
                    "3_Backswing_Total": sj.get("details", {}).get("Backswing", {}).get("Backswing_score", 0),
                    "4_Impact_Total": sj.get("details", {}).get("Impact", {}).get("Impact_score", 0),
                    "5_FollowSwing_Total": sj.get("details", {}).get("FollowSwing", {}).get("FollowSwing_score", 0),
                }
            })

        return {
            "user_id": user_id,
            "range": r,
            "current_sessions": sessions,
            "comparison": _compute_growth_insights(analyses)
        }
    except Exception as e:
        logger_api.exception("User history failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# ⭐ [수정] 사용자 통합 LLM 리포트 생성 API
# ------------------------------------------------------------------
@router.post("/post/user/{user_id}")
def generate_user_llm_report(user_id: str, lang: str = "ko", range: str = "7d", db: Session = Depends(get_db)):
    """사용자의 전체 데이터를 기반으로 AI 종합 리포트를 생성합니다."""
    t0 = time.perf_counter()
    try:
        user_posts = db.query(Post.idx).filter(Post.user_id == user_id).all()
        post_indices = [p.idx for p in user_posts]
        
        analyses = db.query(Analysis).filter(Analysis.post_idx.in_(post_indices)).order_by(Analysis.create_date.asc()).all()
        
        if not analyses:
            raise HTTPException(status_code=404, detail="리포트를 생성할 분석 데이터가 없습니다.")

        # LLM 전달용 통계 데이터 준비
        all_scores = [float((a.score_json or {}).get("total_score", 0)) for a in analyses]
        score_stats = {
            "total_score": {
                "current_mean": round(sum(all_scores) / len(all_scores), 2)
            }
        }
        
        meta = {
            "user_id": user_id,
            "range": range,
            "score_stats": score_stats,
            "insights": _compute_growth_insights(analyses)
        }

        # LLM 호출 (LLM_Total_report.py의 로직 사용)
        report = generate_report(angles={}, meta=meta, lang=lang)

        # ✅ 통합 리포트이므로 가장 최근 post_idx를 연결고리로 저장하거나 별도 테이블 설계 필요
        # 여기서는 가장 최근 분석의 post_idx를 참조로 사용합니다.
        llm_row = LLMReport(
            idx=str(uuid.uuid4()),
            post_idx=analyses[-1].post_idx, 
            feedback=report,
            create_date=datetime.utcnow(),
        )
        db.add(llm_row)
        db.commit()

        return {"report": report, "llm_report_idx": llm_row.idx}

    except Exception as e:
        logger_api.exception("LLM generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))