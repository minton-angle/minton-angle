from typing import Any, Dict, Optional
import logging
import time
import json

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_Total_report import generate_report
from backend.app.services.report.weak_metric_extractor import extract_weak_metrics
from app.services.report.score_stats_service import build_score_report_state
from app.services.report.report_data_service import (
    latest_llm_report_payload,
    load_analysis_windows,
    load_latest_llm_report,
)
from app.services.report.report_generation_service import create_and_save_llm_report
from app.services.report.report_response_service import build_analysis_response

# --- DB/ORM imports for post_idx-based report ---
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
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


def _compute_growth_insights(analyses: list[Analysis]) -> Dict[str, Any]:
    if not analyses:
        return {"growth": None, "plateau": None, "consistency": None, "wins": []}

    keys = ["kf1_error", "kf2_error", "kf3_error"]

    # consistency: std 가장 큰 KF
    stds = {k: _std(_series_vals(analyses, k)) for k in keys}
    volatile = max(stds.items(), key=lambda kv: kv[1])[0] if stds else None

    # plateau: 최근 N mean vs 이전 N mean
    N = min(10, max(4, len(analyses) // 3))
    plateau_obj = None
    plateau_scores = []
    for k in keys:
        arr = _series_vals(analyses, k)
        if len(arr) < 2 * N:
            continue
        prev = arr[-2 * N : -N]
        recent = arr[-N:]
        prev_m = _mean(prev)
        recent_m = _mean(recent)
        delta = recent_m - prev_m
        plateau_scores.append((k, delta, prev_m, recent_m))
    if plateau_scores:
        plateau_scores.sort(key=lambda t: t[1], reverse=True)
        k, delta, prev_m, recent_m = plateau_scores[0]
        plateau_obj = {
            "kf": k,
            "prev_mean": round(prev_m, 4),
            "recent_mean": round(recent_m, 4),
            "delta": round(delta, 4),
            "window_n": N,
        }

    # growth: first vs last
    first_mean = _mean_abs_kf_error(analyses[0])
    last_mean = _mean_abs_kf_error(analyses[-1])
    growth_delta = last_mean - first_mean
    direction = "improved" if growth_delta < -1e-9 else ("worsened" if growth_delta > 1e-9 else "flat")

    # wins: first-half vs second-half delta (가장 개선된 KF top3)
    wins = []
    for k in keys:
        arr = _series_vals(analyses, k)
        if len(arr) < 4:
            continue
        h = max(1, len(arr) // 2)
        m0 = _mean(arr[:h])
        m1 = _mean(arr[h:]) if arr[h:] else m0
        wins.append({"kf": k, "delta": round(m1 - m0, 4), "first_half": round(m0, 4), "second_half": round(m1, 4)})
    wins.sort(key=lambda w: w["delta"])  # most negative = best improvement

    return {
        "growth": {
            "direction": direction,
            "delta_mean_abs_kf_error": round(growth_delta, 4),
            "first_mean_abs_kf_error": round(first_mean, 4),
            "last_mean_abs_kf_error": round(last_mean, 4),
        },
        "plateau": plateau_obj,
        "consistency": {"kf": volatile, "std": round(stds.get(volatile, 0.0), 4)} if volatile else None,
        "wins": wins[:3],
    }

# --------------- POST /api/report/posture ---------------
@router.post("/posture", response_model=PostureReportResponse)
def posture_report(payload: PostureReportRequest):
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/posture start lang=%s meta=%s", payload.lang, (payload.meta or {}))

    try:
        report = generate_report( # 실질적으로 LLM/RAG 파이프라인이 시작되는 위치
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



# --------------- GET /api/report/analysis/post/{post_idx}?range=7d|1m|3m|all ---------------
@router.get("/analysis/post/{post_idx}")
def get_analysis_by_post_alias(
    post_idx: str,
    range: str = "7d",
    db: Session = Depends(get_db),
):
    """
    프론트 차트/테이블 렌더링용 API.
    post_idx 기준으로 Analysis 히스토리를 조회하여
    세션 목록(JSON)으로 반환합니다.

    Query Params:
      - range: "7d" | "1m" | "3m" | "all" | "5n" (default: 7d)
    """
    try:
        # ---- load analyses for current/previous windows ----
        analysis_windows = load_analysis_windows(db=db, post_idx=post_idx, range_value=range)
        r = analysis_windows["range"]
        current_analyses = analysis_windows["current_analyses"]
        prev_analyses = analysis_windows["prev_analyses"]

        latest_llm_payload = latest_llm_report_payload(
            load_latest_llm_report(db=db, post_idx=post_idx)
        )

        payload = build_analysis_response(
            post_idx=post_idx,
            range_value=r,
            current_analyses=current_analyses,
            prev_analyses=prev_analyses,
            latest_llm_payload=latest_llm_payload,
        )

        logger_api.info(
            "[GET ANALYSIS] post_idx=%s range=%s current=%d prev=%d delta=%.4f",
            post_idx,
            r,
            len(payload.get("current_sessions", [])),
            len(payload.get("prev_sessions", [])),
            float(payload.get("comparison", {}).get("delta_mean_abs_kf_error", 0.0)),
        )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger_api.exception("[GET ANALYSIS] failed post_idx=%s err=%s", post_idx, str(e))
        raise HTTPException(status_code=500, detail=str(e))
        
# --------------- POST /api/report/post/{post_idx} ---------------

class PostReportResponse(BaseModel):
    report: Dict[str, Any]


@router.post("/post/{post_idx}")
def posture_report_from_post(
    post_idx: str,
    lang: str = "ko",
    range: str = "7d",
    snapshot_only: bool = False,
    debug_meta: bool = False,
    db: Session = Depends(get_db),
):
    """
    post_idx 기준으로 Analysis 히스토리(여러 건)를 DB에서 조회한 뒤,
        - 최신 1건(latest) + 히스토리 기반 요약지표(insights/추세)만 meta에 포함하여
    - kf1_error/kf2_error/kf3_error 기반 LLM 리포트를 생성합니다.
    """
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/post/%s start lang=%s", post_idx, lang)

    try:
        analysis_windows = load_analysis_windows(db=db, post_idx=post_idx, range_value=range)
        r = analysis_windows["range"]
        analyses = analysis_windows["current_analyses"]
        prev_analyses = analysis_windows["prev_analyses"]

        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx in the selected range")

        latest = analyses[-1]

        # ===== 점수 기반 통계 계산 =====
        # DB에서 조회한 raw Analysis row를 score_stats/trend/kf_stats로 변환한다.
        score_state = build_score_report_state(analyses, prev_analyses)
        score_stats = score_state["score_stats"]
        trend_state = score_state["trend"]
        kf_stats = score_state["kf_stats"]

        insights = _compute_growth_insights(analyses)

        meta: Dict[str, Any] = {
            "post_idx": post_idx,
            "range": r,
            "summary": {
                "current_count": len(analyses),
                "prev_count": len(prev_analyses),
            },
            "trend": trend_state,
            "kf_stats": kf_stats,
            "score_stats": score_stats,
            "insights": insights,
        }

        # score_stats를 LLM reasoning용 movement observation 포맷으로 정규화
        weak_metrics = extract_weak_metrics(score_stats, threshold=90.0)
        meta["weak_metrics"] = weak_metrics

        logger_api.info(
            "[LLM INPUT] weak_metrics=%s",
            json.dumps(weak_metrics, ensure_ascii=False),
        )

        logger_api.info(
            "[LLM INPUT] post_idx=%s range=%s",
            post_idx,
            r,
        )

        logger_api.info(
            "[LLM INPUT] summary=%s",
            json.dumps(meta.get("summary", {}), ensure_ascii=False),
        )
        logger_api.info(
            "[LLM INPUT] score_stats=%s",
            json.dumps(meta.get("score_stats", {}), ensure_ascii=False),
        )
        logger_api.info(
            "[LLM INPUT] trend=%s",
            json.dumps(meta.get("trend", {}), ensure_ascii=False),
        )
        # ---- Snapshot mode (test-only) ----
        # snapshot_only=True 인 경우 LLM 호출/DB저장을 하지 않고 meta만 반환합니다.
        if snapshot_only:
            return {"meta": meta}
        # (2) LLM 호출 및 DB 저장
        result = create_and_save_llm_report(
            db=db,
            post_idx=post_idx,
            meta=meta,
            lang=lang,
        )

        logger_api.info("[LLM SAVE] post_idx=%s llm_report_idx=%s", post_idx, result["llm_report_idx"])

        # (3) 응답
        payload = {
            "report": result["report"],
            "llm_report_idx": result["llm_report_idx"],
        }
        if debug_meta:
            payload["meta"] = meta
        return payload

    except HTTPException:
        raise
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/post/%s failed time_ms=%.1f err=%s", post_idx, dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))