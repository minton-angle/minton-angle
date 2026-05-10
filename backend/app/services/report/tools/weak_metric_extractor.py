from __future__ import annotations

from typing import Any, Dict, List, Optional

"""
[코드 목적]
score_stats에서 sub_score < 90인 세부 metric 목록을 표준 포맷으로 추출하는 기능
"""


TOTAL_TO_STAGE: Dict[str, str] = {
    "1_Ready_Total": "ready",
    "2_Rotation_Total": "rotation",
    "3_Backswing_Total": "backswing",
    "4_Impact_Total": "impact",
}

STAGE_LABELS: Dict[str, str] = {
    "ready": "준비",
    "rotation": "회전",
    "backswing": "백스윙",
    "impact": "임팩트",
    "followswing": "팔로스윙",
}


def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _priority_from_score(score: float) -> str:
    if score < 80:
        return "high"
    if score < 90:
        return "medium"
    return "low"


def _score_band(score: float) -> str:
    if score < 80:
        return "<80"
    if score < 90:
        return "80-90"
    return ">=90"


def _metric_name_from_sub_key(sub_key: str) -> str:
    metric = _safe_str(sub_key)
    if "." in metric:
        metric = metric.split(".")[-1]
    return metric


def extract_weak_metrics(
    score_stats: Dict[str, Any],
    threshold: float = 90.0,
) -> List[Dict[str, Any]]:
    """Normalize low sub-metrics into movement observation records.

    This is deterministic preprocessing, not LLM reasoning.
    It converts router-generated score_stats into a stable input format for
    movement reasoning, RAG query generation, YouTube recommendation, and logs.

    Args:
        score_stats: meta["score_stats"] generated from recent/previous analyses.
        threshold: sub-metric score threshold. Metrics below this value are treated as weak.

    Returns:
        List of weak movement observations sorted by priority and score.
    """
    if not isinstance(score_stats, dict):
        return []

    weak_metrics: List[Dict[str, Any]] = []

    for total_key, stage in TOTAL_TO_STAGE.items():
        stage_node = score_stats.get(total_key) or {}
        if not isinstance(stage_node, dict):
            continue

        stage_current_mean = _safe_float(stage_node.get("current_mean"))
        stage_prev_mean = _safe_float(stage_node.get("prev_mean"))
        stage_delta = _safe_float(stage_node.get("delta"))
        stage_direction = _safe_str(stage_node.get("direction") or "flat")

        sub_stats = stage_node.get("sub_stats") or {}
        if not isinstance(sub_stats, dict):
            continue

        for sub_key, sub_node in sub_stats.items():
            if not isinstance(sub_node, dict):
                continue

            score = _safe_float(sub_node.get("current_mean"))
            if score is None or score >= threshold:
                continue

            prev_score = _safe_float(sub_node.get("prev_mean"))
            delta = _safe_float(sub_node.get("delta"))
            direction = _safe_str(sub_node.get("direction") or stage_direction or "flat")
            metric = _metric_name_from_sub_key(_safe_str(sub_key))

            weak_metrics.append(
                {
                    "stage": stage,
                    "stage_label": STAGE_LABELS.get(stage, stage),
                    "total_key": total_key,
                    "metric": metric,
                    "sub_key": _safe_str(sub_key),
                    "score": round(score, 2),
                    "prev_score": round(prev_score, 2) if prev_score is not None else None,
                    "delta": round(delta, 2) if delta is not None else None,
                    "direction": direction,
                    "priority": _priority_from_score(score),
                    "score_band": _score_band(score),
                    "stage_current_mean": round(stage_current_mean, 2) if stage_current_mean is not None else None,
                    "stage_prev_mean": round(stage_prev_mean, 2) if stage_prev_mean is not None else None,
                    "stage_delta": round(stage_delta, 2) if stage_delta is not None else None,
                    "reasoning_hint": (
                        f"{STAGE_LABELS.get(stage, stage)} 단계의 {metric} 세부 지표가 "
                        f"{round(score, 2)}점으로 기준치 {threshold:g}점 미만입니다."
                    ),
                }
            )

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    weak_metrics.sort(
        key=lambda item: (
            priority_rank.get(_safe_str(item.get("priority")), 9),
            float(item.get("score") or 999),
        )
    )

    return weak_metrics