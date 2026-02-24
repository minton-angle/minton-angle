from typing import Any, Dict, Optional
import logging
import time
import json

import uuid
from datetime import datetime
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_Total_report import generate_report

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
      - range: "7d" | "1m" | "3m" | "all" (default: 7d)
    """
    try:
        r = (range or "7d").lower().strip()
        if r not in ("7d", "1m", "3m", "all"):
            r = "7d"

        now = datetime.utcnow()

        # ---- current window ----
        current_start = None
        if r == "7d":
            current_start = now - timedelta(days=7)
        elif r == "1m":
            current_start = now - timedelta(days=30)
        elif r == "3m":
            current_start = now - timedelta(days=90)
        elif r == "all":
            current_start = None

        base_q = db.query(Analysis).filter(Analysis.post_idx == post_idx)

        # current analyses
        q_current = base_q
        if current_start is not None:
            q_current = q_current.filter(Analysis.create_date >= current_start)

        current_analyses = q_current.order_by(Analysis.create_date.asc()).all()

        if not current_analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx in the selected range")

        # ---- previous window (same length) ----
        prev_analyses = []
        if current_start is not None:
            window_days = (now - current_start).days
            prev_start = current_start - timedelta(days=window_days)
            prev_end = current_start

            q_prev = base_q.filter(
                Analysis.create_date >= prev_start,
                Analysis.create_date < prev_end,
            )
            prev_analyses = q_prev.order_by(Analysis.create_date.asc()).all()

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
        latest_llm = (
            db.query(LLMReport)
            .filter(LLMReport.post_idx == post_idx)
            .order_by(LLMReport.create_date.desc())
            .first()
        )

        latest_llm_payload = None
        if latest_llm is not None:
            latest_llm_payload = {
                "idx": latest_llm.idx,
                "created_at": latest_llm.create_date.isoformat() if latest_llm.create_date else None,
                "report": latest_llm.feedback,
            }

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


@router.post("/post/{post_idx}", response_model=PostReportResponse)
def posture_report_from_post(
    post_idx: str,
    lang: str = "ko",
    range: str = "7d",
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
        r = (range or "7d").lower().strip()
        if r not in ("7d", "1m", "3m", "all"):
            r = "7d"

        now = datetime.utcnow()
        base_q = db.query(Analysis).filter(Analysis.post_idx == post_idx)

        current_start = None
        if r == "7d":
            current_start = now - timedelta(days=7)
        elif r == "1m":
            current_start = now - timedelta(days=30)
        elif r == "3m":
            current_start = now - timedelta(days=90)
        elif r == "all":
            current_start = None

        q_current = base_q
        if current_start is not None:
            q_current = q_current.filter(Analysis.create_date >= current_start)

        analyses = q_current.order_by(Analysis.create_date.asc()).all()

        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx in the selected range")

        # previous window (same length)
        prev_analyses = []
        if current_start is not None:
            window_days = (now - current_start).days
            prev_start = current_start - timedelta(days=window_days)
            prev_end = current_start

            q_prev = base_q.filter(
                Analysis.create_date >= prev_start,
                Analysis.create_date < prev_end,
            )
            prev_analyses = q_prev.order_by(Analysis.create_date.asc()).all()

        latest = analyses[-1]

        # 점수 기반 리포트로 전환: angles(단일 세션)은 사용하지 않음
        angles = {}

        # ===== 점수 기반: Stage score들 + total_score 통계 =====
        SCORE_KEYS = [
            "1_Ready_Total",
            "2_Rotation_Total",
            "3_Backswing_Total",
            "4_Impact_Total",
            "5_FollowSwing_SuccessRate",
            "total_score",
        ]

        # ===== stage breakdown(세부 항목) 정의 =====
        # 새로운 score_json 스키마(details 기반)에서 stage별 세부 metric명을 동적으로 수집합니다.
        STAGES = {
            "1_Ready_Total": "Ready",
            "2_Rotation_Total": "Rotation",
            "3_Backswing_Total": "Backswing",
            "4_Impact_Total": "Impact",
        }

        def _details_node(row: Analysis) -> Dict[str, Any]:
            sj = getattr(row, "score_json", None) or {}
            details = sj.get("details") if isinstance(sj, dict) else None
            return details if isinstance(details, dict) else {}

        def _stage_node(row: Analysis, stage_name: str) -> Dict[str, Any]:
            details = _details_node(row)
            node = details.get(stage_name)
            return node if isinstance(node, dict) else {}

        def _metric_score(row: Analysis, stage_name: str, metric_name: str) -> Optional[float]:
            node = _stage_node(row, stage_name)
            metric = node.get(metric_name)
            if not isinstance(metric, dict):
                return None
            v = metric.get("score")
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        def _stage_score(row: Analysis, stage_name: str) -> Optional[float]:
            node = _stage_node(row, stage_name)
            v = node.get(f"{stage_name}_score")
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        def _mean_metric_score(rows: list[Analysis], stage_name: str, metric_name: str) -> float:
            vals: list[float] = []
            for rr in rows:
                v = _metric_score(rr, stage_name, metric_name)
                if v is None:
                    continue
                v = max(0.0, min(100.0, float(v)))
                vals.append(v)
            return sum(vals) / len(vals) if vals else 0.0

        def _collect_stage_metrics(rows: list[Analysis], stage_name: str) -> list[str]:
            # rows에서 stage_name 아래의 metric key들을 수집( *_score 제외 )
            keys: set[str] = set()
            for rr in rows:
                node = _stage_node(rr, stage_name)
                for k, v in node.items():
                    if k == f"{stage_name}_score":
                        continue
                    # FollowSwing의 경우 Performance는 stage breakdown에 포함하지 않음(별도 성공률 처리)
                    if stage_name == "FollowSwing" and k == "Performance":
                        continue
                    if isinstance(v, dict) and "score" in v:
                        keys.add(str(k))
            return sorted(keys)

        def _compute_breakdown_stats(total_key: str, stage_name: str, cur_rows: list[Analysis], prev_rows: list[Analysis]) -> Dict[str, Any]:
            """각 Stage Total에 대해 세부 항목 통계 + worst_sub(가장 낮은 세부 점수) 요약을 생성."""
            sub_keys = _collect_stage_metrics(cur_rows, stage_name)
            sub_stats: Dict[str, Any] = {}
            for sk in sub_keys:
                cm = _mean_metric_score(cur_rows, stage_name, sk)
                pm = _mean_metric_score(prev_rows, stage_name, sk) if prev_rows else cm
                d = cm - pm
                sd = "improved" if d > 1e-9 else ("worsened" if d < -1e-9 else "flat")
                # metric id는 RAG/KB 매칭을 위해 stage 접두를 붙여 저장
                metric_id = f"{stage_name}.{sk}"
                sub_stats[metric_id] = {
                    "current_mean": round(cm, 2),
                    "prev_mean": round(pm, 2),
                    "delta": round(d, 2),
                    "direction": sd,
                }

            worst_sub = None
            worst_val = None
            for sk, node in sub_stats.items():
                v = node.get("current_mean")
                try:
                    v = float(v)
                except Exception:
                    continue
                if worst_val is None or v < worst_val:
                    worst_val = v
                    worst_sub = sk

            return {
                "sub_stats": sub_stats,
                "worst_sub": worst_sub,
                "worst_sub_current_mean": (round(float(worst_val), 2) if worst_val is not None else None),
            }

        def _score_of(row: Analysis, key: str) -> Optional[float]:
            sj = getattr(row, "score_json", None) or {}
            if not isinstance(sj, dict):
                return None

            # total_score
            if key == "total_score":
                v = sj.get("total_score")
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            # stage totals
            stage_name = STAGES.get(key)
            if stage_name:
                v = _stage_score(row, stage_name)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            return None

        def _mean_score(rows: list[Analysis], key: str) -> float:
            vals: list[float] = []
            for r in rows:
                v = _score_of(r, key)
                if v is None:
                    continue
                v = max(0.0, min(100.0, float(v)))
                vals.append(v)
            return sum(vals) / len(vals) if vals else 0.0

        # 팔로스윙 false rate / risk level 계산 함수 추가
        def _followswing_false_rate(rows: list[Analysis]) -> float:
            total = 0
            false_n = 0
            for rr in rows:
                sj = getattr(rr, "score_json", None) or {}
                details = sj.get("details") if isinstance(sj, dict) else None
                if not isinstance(details, dict):
                    continue
                fs = details.get("FollowSwing")
                if not isinstance(fs, dict):
                    continue
                perf = fs.get("Performance")
                if not isinstance(perf, dict):
                    continue
                passed = perf.get("success")
                if passed is not True and passed is not False:
                    continue

                total += 1
                if not passed:
                    false_n += 1

            return (false_n / total) if total else 0.0

        def _followswing_risk_level(false_rate: float) -> str:
            # <40%: ok, 40~<80%: improve, >=80%: risk
            if false_rate >= 0.80:
                return "risk"
            if false_rate >= 0.40:
                return "improve"
            return "ok"

        score_stats: Dict[str, Any] = {}

        for k in SCORE_KEYS:
            # FollowSwing: 성공률(=100 - false%) 기반으로 score_stats 구성
            if k == "5_FollowSwing_SuccessRate":
                cur_false = _followswing_false_rate(analyses)
                prev_false = _followswing_false_rate(prev_analyses) if prev_analyses else cur_false

                cur_sr = 100.0 - (cur_false * 100.0)
                prev_sr = 100.0 - (prev_false * 100.0)

                dlt = cur_sr - prev_sr
                direction = "improved" if dlt > 1e-9 else ("worsened" if dlt < -1e-9 else "flat")

                score_stats[k] = {
                    "current_mean": round(cur_sr, 2),
                    "prev_mean": round(prev_sr, 2),
                    "delta": round(dlt, 2),
                    "direction": direction,

                    # LLM에게만 제공할 위험 신호(숫자/레벨)
                    "false_rate_current": round(cur_false, 4),
                    "false_rate_prev": round(prev_false, 4),
                    "risk_level": _followswing_risk_level(cur_false),
                    "success_rate_current": round(cur_sr, 2),
                    "success_rate_prev": round(prev_sr, 2)
                }
                continue

            # (나머지 키는 기존 점수 평균 로직 그대로)
            cur_m = _mean_score(analyses, k)
            prev_m = _mean_score(prev_analyses, k) if prev_analyses else cur_m
            dlt = cur_m - prev_m
            direction = "improved" if dlt > 1e-9 else ("worsened" if dlt < -1e-9 else "flat")
            node: Dict[str, Any] = {
                "current_mean": round(cur_m, 2),
                "prev_mean": round(prev_m, 2),
                "delta": round(dlt, 2),
                "direction": direction,
            }

            # Total 키는 세부 항목 breakdown 요약을 추가(LLM 문장 다양성 목적)
            if k in STAGES:
                node.update(_compute_breakdown_stats(k, STAGES[k], analyses, prev_analyses))

            score_stats[k] = node

        # 전체 트렌드는 total_score 기준으로 요약
        cur_avg = float(score_stats.get("total_score", {}).get("current_mean", 0.0))
        prev_avg = float(score_stats.get("total_score", {}).get("prev_mean", cur_avg))
        avg_delta = round(cur_avg - prev_avg, 2)
        trend = "improved" if avg_delta > 1e-9 else ("worsened" if avg_delta < -1e-9 else "flat")

        # 점수 기반 리포트에서는 kf_stats 대신 score_stats 사용
        kf_stats = {}

        # ===== KF별 평균 비교 =====
        def _mean_of(rows, attr):
            if not rows:
                return 0.0
            vals = [float(getattr(r, attr) or 0.0) for r in rows]
            return sum(vals) / len(vals)

        kf_stats = {}
        for key in ["kf1_error", "kf2_error", "kf3_error"]:
            cur_kf_mean = _mean_of(analyses, key)
            prev_kf_mean = _mean_of(prev_analyses, key) if prev_analyses else cur_kf_mean
            kf_delta = cur_kf_mean - prev_kf_mean
            kf_direction = "improved" if kf_delta < -1e-9 else ("worsened" if kf_delta > 1e-9 else "flat")

            kf_stats[key] = {
                "current_mean": round(cur_kf_mean, 4),
                "prev_mean": round(prev_kf_mean, 4),
                "delta": round(kf_delta, 4),
                "direction": kf_direction,
            }

        insights = _compute_growth_insights(analyses)

        meta: Dict[str, Any] = {
            "post_idx": post_idx,
            "range": r,
            "summary": {
                "current_count": len(analyses),
                "prev_count": len(prev_analyses),
            },
            "trend": {
                "current_mean_average_score": round(cur_avg, 2),
                "prev_mean_average_score": round(prev_avg, 2),
                "delta_average_score": avg_delta,
                "direction": trend,
            },
            "score_stats": score_stats,
            "insights": insights,
        }

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
        return {"report": report, "llm_report_idx": llm_row.idx}

    except HTTPException:
        raise
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/post/%s failed time_ms=%.1f err=%s", post_idx, dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))