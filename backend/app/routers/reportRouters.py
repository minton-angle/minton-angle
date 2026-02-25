from typing import Any, Dict, Optional
import logging
import time

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_report import generate_report

# --- DB/ORM imports for post_idx-based report ---
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.postModels import Post
from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport

from app.models.fileModels import File


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


# ⭐ 경로 변환 유틸 함수
def fix_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    clean = raw_path.replace("\\", "/")
    # realtime 경로
    for marker in ["data/realtime/", "data/upload/"]:
        idx = clean.find(marker)
        if idx != -1:
            return "/data/" + clean[idx + len("data/"):]
    # backend/data 패턴
    marker = "backend/data/"
    idx = clean.find(marker)
    if idx != -1:
        return "/" + clean[idx:]
    return clean


# ⭐ 파일 타입 → 프론트 키 매핑
FILE_TYPE_MAP = {
    "READY":           "ready",        # ⭐ kf1_image → ready
    "SEQ1_READY":      "seq1_ready",
    "SEQ2_TAKEAWAY":   "seq2_takeaway",
    "SEQ3_BACKSWING":  "seq3_backswing",
    "SEQ4_DOWNSWING1": "seq4_downswing1",
    "SEQ5_DOWNSWING2": "seq5_downswing2",
    "SEQ6_IMPACT":     "seq6_impact",
    "IMPACT":          "impact",       # ⭐ kf3_image → impact
    "FOLLOWSWING":     "followswing",  # ⭐ follow_video → followswing
}


# --------------- POST /api/report/posture ---------------
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
                "frame": "ALL",
                "score": score,
                "kf_error": round(mean_err, 4),
            })

        logger_api.info("[GET ANALYSIS] post_idx=%s row_count=%d", post_idx, len(sessions))

        return {
            "post_idx": post_idx,
            "sessions": sessions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger_api.exception("[GET ANALYSIS] failed post_idx=%s err=%s", post_idx, str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --------------- POST /api/report/post/{post_idx} ---------------

class PostReportResponse(BaseModel):
    report: Dict[str, Any]


@router.post("/post/{post_idx}", response_model=PostReportResponse)
def posture_report_from_post(post_idx: str, lang: str = "ko", db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/post/%s start lang=%s", post_idx, lang)

    try:
        analyses = (
            db.query(Analysis)
            .filter(Analysis.post_idx == post_idx)
            .order_by(Analysis.create_date.asc())
            .all()
        )
        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx")

        latest = analyses[-1]
        first = analyses[0]

        angles = {
            "kf1_error": float(latest.kf1_error or 0.0),
            "kf2_error": float(latest.kf2_error or 0.0),
            "kf3_error": float(latest.kf3_error or 0.0),
        }

        first_mean = _mean_abs_kf_error(first)
        last_mean = _mean_abs_kf_error(latest)
        delta = last_mean - first_mean
        trend = "improved" if delta < -1e-9 else ("worsened" if delta > 1e-9 else "flat")

        history = [
            {
                "idx": a.idx,
                "created_at": a.create_date.isoformat() if a.create_date else None,
                "kf1_error": float(a.kf1_error or 0.0),
                "kf2_error": float(a.kf2_error or 0.0),
                "kf3_error": float(a.kf3_error or 0.0),
                "mean_abs_kf_error": round(_mean_abs_kf_error(a), 4),
                "score_json": a.score_json or {},
            }
            for a in analyses
        ]

        meta: Dict[str, Any] = {
            "post_idx": post_idx,
            "latest": {
                "idx": latest.idx,
                "created_at": latest.create_date.isoformat() if latest.create_date else None,
                "kf1": latest.kf1,
                "kf2": latest.kf2,
                "kf3": latest.kf3,
                "score_json": latest.score_json or {},
            },
            "history": history,
            "trend": {
                "first_at": first.create_date.isoformat() if first.create_date else None,
                "last_at": latest.create_date.isoformat() if latest.create_date else None,
                "first_mean_abs_kf_error": round(first_mean, 4),
                "last_mean_abs_kf_error": round(last_mean, 4),
                "delta_mean_abs_kf_error": round(delta, 4),
                "direction": trend,
            },
        }

        report = generate_report(angles=angles, meta=meta, lang=lang)
        llm_row = LLMReport(
            idx=str(uuid.uuid4()),
            post_idx=post_idx,
            feedback=report,
            create_date=datetime.utcnow(),
        )
        db.add(llm_row)
        db.commit()

        return {"report": report, "llm_report_idx": llm_row.idx}

    except HTTPException:
        raise
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/post/%s failed time_ms=%.1f err=%s", post_idx, dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------- GET /api/report/upload/result/{post_idx} ---------------
@router.get("/upload/result/{post_idx}")
async def get_analysis_result(
    post_idx: str,
    db: Session = Depends(get_db)
):
    try:
        post = db.query(Post).filter(Post.idx == post_idx).first()
        if not post:
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")

        # ⭐ 실시간 분석 (3회 스윙)
        if post.type == "REALTIME":
            analyses = db.query(Analysis).filter(
                Analysis.post_idx == post_idx
            ).order_by(Analysis.swing_num).all()

            if not analyses:
                raise HTTPException(status_code=404, detail="분석 결과가 없습니다")

            all_files = db.query(File).filter(File.post_idx == post_idx).all()

            swings = {}
            for analysis in analyses:
                swing_num = analysis.swing_num
                swing_files = [f for f in all_files if f.swing_num == swing_num]

                # ⭐ 파일 타입 매핑 + 경로 변환
                files_dict = {}
                for file in swing_files:
                    key = FILE_TYPE_MAP.get(file.file_type)
                    if key:
                        files_dict[key] = fix_path(file.file_path)

                score_data = analysis.score_json or {}

                swings[str(swing_num)] = {
                    "total_score": score_data.get('total_score', 0),
                    "kf1": analysis.kf1,
                    "kf2": analysis.kf2,
                    "kf3": analysis.kf3,
                    "scores": score_data,
                    "files": files_dict
                }

            logger_api.info(
                "[GET RESULT] post_idx=%s type=REALTIME swing_count=%d",
                post_idx, len(swings)
            )

            return {
                "success": True,
                "type": "realtime",
                "swings": swings
            }

        # ⭐ 동영상 업로드 (단일 분석)
        else:
            analysis = db.query(Analysis).filter(
                Analysis.post_idx == post_idx
            ).first()

            if not analysis:
                raise HTTPException(status_code=404, detail="분석 결과가 없습니다")

            all_files = db.query(File).filter(File.post_idx == post_idx).all()

            # ⭐ 파일 타입 매핑 + 경로 변환
            files_dict = {}
            for file in all_files:
                key = FILE_TYPE_MAP.get(file.file_type)
                if key:
                    files_dict[key] = fix_path(file.file_path)

            score_data = analysis.score_json or {}

            logger_api.info("[GET RESULT] post_idx=%s type=VIDEO", post_idx)

            return {
                "success": True,
                "type": "video",
                "total_score": score_data.get('total_score', 0),
                "scores": score_data,
                "files": files_dict
            }

    except HTTPException:
        raise
    except Exception as e:
        logger_api.exception("[GET RESULT] failed post_idx=%s err=%s", post_idx, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime/result/{post_idx}")
async def get_realtime_result(post_idx: str, db: Session = Depends(get_db)):
    """실시간 분석 결과 조회 (3회 스윙)"""
    
    post = db.query(Post).filter(Post.idx == post_idx).first()
    if not post:
        raise HTTPException(status_code=404, detail="POST를 찾을 수 없습니다.")
    
    analyses = db.query(Analysis).filter(
        Analysis.post_idx == post_idx
    ).order_by(Analysis.swing_num).all()
    
    swings = {}
    
    for analysis in analyses:
        swing_num = analysis.swing_num
        
        swing_files = db.query(File).filter(
            File.post_idx == post_idx,
            File.swing_num == swing_num
        ).all()
        
        file_paths = {}
        for f in swing_files:
            key = FILE_TYPE_MAP.get(f.file_type)  # ⭐ lower() 대신 FILE_TYPE_MAP 사용
            if key:
                file_paths[key] = fix_path(f.file_path)
        
        swings[str(swing_num)] = {
            "swing_num": swing_num,
            "total_score": analysis.score_json.get('total_score', 0) if analysis.score_json else 0,
            "scores": analysis.score_json,
            "files": file_paths
        }
    
    return {
        "success": True,
        "post_idx": post_idx,
        "type": "realtime",
        "total_score": post.total_score,
        "swings": swings
    }