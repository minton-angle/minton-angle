from __future__ import annotations

from typing import Any, Dict, Optional

from app.models.analysisModels import Analysis


def _clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, x))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stage_score(score_json: Dict[str, Any], stage: str) -> Optional[float]:
    details = score_json.get("details") if isinstance(score_json, dict) else None
    if not isinstance(details, dict):
        return None
    node = details.get(stage)
    if not isinstance(node, dict):
        return None
    value = node.get(f"{stage}_score")
    try:
        return _clamp_score(value) if value is not None else None
    except Exception:
        return None


def _followswing_success(score_json: Dict[str, Any]) -> Optional[bool]:
    details = score_json.get("details") if isinstance(score_json, dict) else None
    if not isinstance(details, dict):
        return None
    fs = details.get("FollowSwing")
    if not isinstance(fs, dict):
        return None
    perf = fs.get("Performance")
    if not isinstance(perf, dict):
        return None
    value = perf.get("success")
    if value is True:
        return True
    if value is False:
        return False
    return None


def analysis_to_sessions(analyses: list[Analysis]) -> list[Dict[str, Any]]:
    """Serialize Analysis rows into frontend session objects."""
    sessions: list[Dict[str, Any]] = []

    for row in analyses:
        score_json = getattr(row, "score_json", None) or {}

        stage_scores = {
            "1_Ready_Total": _clamp_score(_stage_score(score_json, "Ready") or 0.0),
            "2_Rotation_Total": _clamp_score(_stage_score(score_json, "Rotation") or 0.0),
            "3_Backswing_Total": _clamp_score(_stage_score(score_json, "Backswing") or 0.0),
            "4_Impact_Total": _clamp_score(_stage_score(score_json, "Impact") or 0.0),
            "5_FollowSwing_Total": _clamp_score(_stage_score(score_json, "FollowSwing") or 0.0),
        }

        total_score = score_json.get("total_score") if isinstance(score_json, dict) else None
        total_score_num = None
        try:
            if total_score is not None:
                total_score_num = _clamp_score(total_score)
        except Exception:
            total_score_num = None

        session_score = int(round(total_score_num)) if total_score_num is not None else 0
        followswing_pass = _followswing_success(score_json)

        sessions.append(
            {
                "idx": row.idx,
                "created_at": row.create_date.isoformat() if row.create_date else None,
                "frame": "ALL",
                "score": session_score,
                "stage_scores": stage_scores,
                "total_score": total_score_num,
                "followswing_pass": followswing_pass,
            }
        )

    return sessions


def _mean_average_score(sessions: list[Dict[str, Any]]) -> float:
    vals = [float(session.get("average_score") or session.get("score") or 0.0) for session in sessions]
    return _mean(vals)


def build_analysis_comparison(
    current_sessions: list[Dict[str, Any]],
    prev_sessions: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build current/previous score comparison summary for the analysis GET API."""
    current_avg = _mean_average_score(current_sessions)
    prev_avg = _mean_average_score(prev_sessions)
    avg_delta = round(current_avg - prev_avg, 2)
    score_direction = "improved" if avg_delta > 1e-9 else ("worsened" if avg_delta < -1e-9 else "flat")

    return {
        "current_mean_average_score": round(current_avg, 2),
        "prev_mean_average_score": round(prev_avg, 2),
        "delta_average_score": avg_delta,
        "score_direction": score_direction,
    }


def build_analysis_response(
    *,
    post_idx: str,
    range_value: str,
    current_analyses: list[Analysis],
    prev_analyses: list[Analysis],
    latest_llm_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build full GET /analysis/post/{post_idx} response payload."""
    current_sessions = analysis_to_sessions(current_analyses)
    prev_sessions = analysis_to_sessions(prev_analyses)
    comparison = build_analysis_comparison(current_sessions, prev_sessions)

    return {
        "post_idx": post_idx,
        "range": range_value,
        "current_sessions": current_sessions,
        "prev_sessions": prev_sessions,
        "latest_llm_report": latest_llm_payload,
        "comparison": comparison,
    }