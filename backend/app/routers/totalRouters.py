from typing import Any, Dict, Optional
import logging
import time
import json

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_Total_report import generate_report
from app.services.report.tools.weak_metric_extractor import extract_weak_metrics
from app.services.report.score_stats_service import build_score_report_state
from app.services.report.report_data_service import (
    latest_llm_report_payload,
    load_analysis_windows,
    load_latest_llm_report,
)

# --- DB/ORM imports for post_idx-based report ---
from sqlalchemy.orm import Session

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

def _series_vals(analyses: list[Analysis], key: str) -> list[float]:
    out: list[float] = []
    for a in analyses:
        v = getattr(a, key, None)
        if v is None:
            continue
        try:
            out.append(abs(float(v)))
        except Exception:
            continue
    return out

def _mean(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0

def _std(arr: list[float]) -> float:
    if not arr:
        return 0.0
    m = _mean(arr)
    return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5

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

        def to_sessions(analyses: list[Analysis]) -> list[Dict[str, Any]]:
            def _clamp_score(v: Any) -> float:
                try:
                    x = float(v)
                except Exception:
                    return 0.0
                return max(0.0, min(100.0, x))

            def _stage_score(sj: Dict[str, Any], stage: str) -> Optional[float]:
                details = sj.get("details") if isinstance(sj, dict) else None
                if not isinstance(details, dict):
                    return None
                node = details.get(stage)
                if not isinstance(node, dict):
                    return None
                v = node.get(f"{stage}_score")
                try:
                    return _clamp_score(v) if v is not None else None
                except Exception:
                    return None

            def _followswing_success(sj: Dict[str, Any]) -> Optional[bool]:
                details = sj.get("details") if isinstance(sj, dict) else None
                if not isinstance(details, dict):
                    return None
                fs = details.get("FollowSwing")
                if not isinstance(fs, dict):
                    return None
                perf = fs.get("Performance")
                if not isinstance(perf, dict):
                    return None
                v = perf.get("success")
                if v is True:
                    return True
                if v is False:
                    return False
                return None

            out = []
            for a in analyses:
                # legacy KF-based score
                mean_err = _mean_abs_kf_error(a)

                sj = getattr(a, "score_json", None) or {}

                # stage mean scores (for charts/cards)
                stage_scores = {
                    "1_Ready_Total": _clamp_score(_stage_score(sj, "Ready") or 0.0),
                    "2_Rotation_Total": _clamp_score(_stage_score(sj, "Rotation") or 0.0),
                    "3_Backswing_Total": _clamp_score(_stage_score(sj, "Backswing") or 0.0),
                    "4_Impact_Total": _clamp_score(_stage_score(sj, "Impact") or 0.0),
                    # FollowSwing is still rendered as a stage score for UI; use stage score if present, else 0
                    "5_FollowSwing_Total": _clamp_score(_stage_score(sj, "FollowSwing") or 0.0),
                }

                # total score for ring/chart (strictly total_score)
                total_score = sj.get("total_score", None)
                total_score_num = None
                try:
                    if total_score is not None:
                        total_score_num = _clamp_score(total_score)
                except Exception:
                    total_score_num = None

                session_score = int(round(total_score_num)) if total_score_num is not None else 0

                # boolean followswing pass for donut/false-rate logic on FE
                followswing_pass = _followswing_success(sj)

                out.append({
                    "idx": a.idx,
                    "created_at": a.create_date.isoformat() if a.create_date else None,
                    "frame": "ALL",

                    # score for ring/chart (strictly total_score)
                    "score": session_score,

                    # legacy fields (keep)
                    "kf_error": round(mean_err, 4),
                    "kf1_error": float(a.kf1_error or 0.0),
                    "kf2_error": float(a.kf2_error or 0.0),
                    "kf3_error": float(a.kf3_error or 0.0),

                    # new fields
                    "stage_scores": stage_scores,
                    "total_score": total_score_num,
                    "followswing_pass": followswing_pass,
                })
            return out

        current_sessions = to_sessions(current_analyses)
        prev_sessions = to_sessions(prev_analyses)

        # ---- latest LLM report (optional) ----
        latest_llm_payload = latest_llm_report_payload(
            load_latest_llm_report(db=db, post_idx=post_idx)
        )

        # ---- summary stats ----
        def mean_kf(arr: list[Dict[str, Any]]) -> float:
            vals = [abs(float(s.get("kf_error", 0.0))) for s in arr]
            return sum(vals) / len(vals) if vals else 0.0

        current_mean = mean_kf(current_sessions)
        prev_mean = mean_kf(prev_sessions)
        delta = round(current_mean - prev_mean, 4)

        direction = "improved" if delta < 0 else ("worsened" if delta > 0 else "flat")

        logger_api.info(
            "[GET ANALYSIS] post_idx=%s range=%s current=%d prev=%d delta=%.4f",
            post_idx,
            r,
            len(current_sessions),
            len(prev_sessions),
            delta,
        )

        return {
            "post_idx": post_idx,
            "range": r,
            "current_sessions": current_sessions,
            "prev_sessions": prev_sessions,
            "latest_llm_report": latest_llm_payload,
            "comparison": {
                "current_mean_abs_kf_error": round(current_mean, 4),
                "prev_mean_abs_kf_error": round(prev_mean, 4),
                "delta_mean_abs_kf_error": delta,
                "direction": direction,

                # score-based (Average_Score) comparison for UI
                "current_mean_average_score": round(
                    sum([float(s.get("average_score") or s.get("score") or 0.0) for s in current_sessions]) / max(1, len(current_sessions)),
                    2,
                ),
                "prev_mean_average_score": round(
                    sum([float(s.get("average_score") or s.get("score") or 0.0) for s in prev_sessions]) / max(1, len(prev_sessions)) if prev_sessions else 0.0,
                    2,
                ),
                "delta_average_score": round(
                    (sum([float(s.get("average_score") or s.get("score") or 0.0) for s in current_sessions]) / max(1, len(current_sessions))) -
                    (sum([float(s.get("average_score") or s.get("score") or 0.0) for s in prev_sessions]) / max(1, len(prev_sessions)) if prev_sessions else 0.0),
                    2,
                ),
                "score_direction": (
                    "improved" if ((sum([float(s.get("average_score") or s.get("score") or 0.0) for s in current_sessions]) / max(1, len(current_sessions))) -
                                   (sum([float(s.get("average_score") or s.get("score") or 0.0) for s in prev_sessions]) / max(1, len(prev_sessions)) if prev_sessions else 0.0)) > 1e-9
                    else ("worsened" if ((sum([float(s.get("average_score") or s.get("score") or 0.0) for s in current_sessions]) / max(1, len(current_sessions))) -
                                         (sum([float(s.get("average_score") or s.get("score") or 0.0) for s in prev_sessions]) / max(1, len(prev_sessions)) if prev_sessions else 0.0)) < -1e-9
                          else "flat")
                ),
            },
        }

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

        # 점수 기반 리포트로 전환: angles(단일 세션)은 사용하지 않음
        angles = {}

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
        # (2) LLM 호출
        report = generate_report(angles=angles, meta=meta, lang=lang)
        # (3) ✅ DB 저장 (llm_report)
        llm_row = LLMReport(
            idx=str(uuid.uuid4()),
            post_idx=post_idx,
            feedback=report,                 # ← JSON 그대로 저장
            create_date=datetime.utcnow(),
        )
        db.add(llm_row)
        db.commit()

        logger_api.info("[LLM SAVE] post_idx=%s llm_report_idx=%s", post_idx, llm_row.idx)

        # (4) 응답
        payload = {"report": report, "llm_report_idx": llm_row.idx}
        if debug_meta:
            payload["meta"] = meta
        return payload

    except HTTPException:
        raise
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/post/%s failed time_ms=%.1f err=%s", post_idx, dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))