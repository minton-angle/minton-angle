from typing import Any, Dict, Optional
import logging
import time


from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_report import generate_report

# --- DB/ORM imports for post_idx-based report ---
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.postModels import Post
from app.models.analysisModels import Analysis



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(prefix="/api/report", tags=["Report"])


logger_api = logging.getLogger("app.api")

class PostureReportRequest(BaseModel):
    angles: Dict[str, float] = Field(..., description="관절별 오차각도")
    meta: Optional[Dict[str, Any]] = Field(default=None)
    lang: str = Field(default="ko")


class PostureReportResponse(BaseModel):
    report: Dict[str, Any]

def _mean_abs_kf_error(a: Analysis) -> float:
    vals = []
    for v in (a.kf1_error, a.kf2_error, a.kf3_error):
        try:
            if v is None:
                continue
            vals.append(abs(float(v)))
        except Exception:
            continue
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
    
@router.post("/posture", response_model=PostureReportResponse)
def posture_report(payload: PostureReportRequest):
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/posture start lang=%s meta=%s", payload.lang, (payload.meta or {}))

    try:
        report = generate_report(
            angles=payload.angles,
            meta=payload.meta,
            lang=payload.lang,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.info(
            "POST /api/report/posture ok time_ms=%.1f severity=%s summary=%s",
            dt_ms,
            report.get("overall_severity"),
            (report.get("summary") or "")[:120],
        )
        return {"report": report}
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/posture failed time_ms=%.1f err=%s", dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))



# --------------- GET /api/report/analysis/post/{post_idx} ---------------
@router.get("/analysis/post/{post_idx}")
def get_analysis_by_post_alias(post_idx: str, db: Session = Depends(get_db)):
    """
    프론트 차트/테이블 렌더링용 API.
    post_idx 기준으로 Analysis 히스토리를 조회하여
    세션 목록(JSON)으로 반환합니다.
    """
    try:
        analyses = (
            db.query(Analysis)
            .filter(Analysis.post_idx == post_idx)
            .order_by(Analysis.create_date.asc())
            .all()
        )

        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx")

        sessions = []
        for a in analyses:
            mean_err = _mean_abs_kf_error(a)
            score = round(max(0, min(100, 100 - (mean_err / 20) * 100)))

            sessions.append({
                "idx": a.idx,
                "created_at": a.create_date.isoformat() if a.create_date else None,
                "frame": "ALL",  # 종합 페이지 기준
                "score": score,
                "kf_error": round(mean_err, 4),
            })

        logger_api.info(
            "[GET ANALYSIS] post_idx=%s row_count=%d",
            post_idx,
            len(sessions),
        )

        return {
            "post_idx": post_idx,
            "sessions": sessions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger_api.exception("[GET ANALYSIS] failed post_idx=%s err=%s", post_idx, str(e))
        raise HTTPException(status_code=500, detail=str(e))

