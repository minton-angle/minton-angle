from __future__ import annotations

from typing import Any, Dict, Optional

from app.models.analysisModels import Analysis


SCORE_KEYS = [
    "1_Ready_Total",
    "2_Rotation_Total",
    "3_Backswing_Total",
    "4_Impact_Total",
    "5_FollowSwing_SuccessRate",
    "total_score",
]

STAGES: Dict[str, str] = {
    "1_Ready_Total": "Ready",
    "2_Rotation_Total": "Rotation",
    "3_Backswing_Total": "Backswing",
    "4_Impact_Total": "Impact",
}


def _clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, x))


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


def _score_of(row: Analysis, key: str) -> Optional[float]:
    sj = getattr(row, "score_json", None) or {}
    if not isinstance(sj, dict):
        return None

    if key == "total_score":
        v = sj.get("total_score")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    stage_name = STAGES.get(key)
    if stage_name:
        return _stage_score(row, stage_name)

    return None


def _mean_score(rows: list[Analysis], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        v = _score_of(row, key)
        if v is None:
            continue
        vals.append(_clamp_score(v))
    return sum(vals) / len(vals) if vals else 0.0


def _mean_metric_score(rows: list[Analysis], stage_name: str, metric_name: str) -> float:
    vals: list[float] = []
    for row in rows:
        v = _metric_score(row, stage_name, metric_name)
        if v is None:
            continue
        vals.append(_clamp_score(v))
    return sum(vals) / len(vals) if vals else 0.0


def _collect_stage_metrics(rows: list[Analysis], stage_name: str) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        node = _stage_node(row, stage_name)
        for key, value in node.items():
            if key == f"{stage_name}_score":
                continue
            if stage_name == "FollowSwing" and key == "Performance":
                continue
            if isinstance(value, dict) and "score" in value:
                keys.add(str(key))
    return sorted(keys)


def _compute_breakdown_stats(
    stage_name: str,
    cur_rows: list[Analysis],
    prev_rows: list[Analysis],
) -> Dict[str, Any]:
    """Create sub-metric stats and weakest sub-metric summary for one stage."""
    sub_keys = _collect_stage_metrics(cur_rows, stage_name)
    sub_stats: Dict[str, Any] = {}

    for sub_key in sub_keys:
        current_mean = _mean_metric_score(cur_rows, stage_name, sub_key)
        prev_mean = _mean_metric_score(prev_rows, stage_name, sub_key) if prev_rows else current_mean
        delta = current_mean - prev_mean
        direction = "improved" if delta > 1e-9 else ("worsened" if delta < -1e-9 else "flat")
        metric_id = f"{stage_name}.{sub_key}"

        sub_stats[metric_id] = {
            "current_mean": round(current_mean, 2),
            "prev_mean": round(prev_mean, 2),
            "delta": round(delta, 2),
            "direction": direction,
        }

    worst_sub = None
    worst_val = None
    for sub_key, node in sub_stats.items():
        try:
            value = float(node.get("current_mean"))
        except Exception:
            continue
        if worst_val is None or value < worst_val:
            worst_val = value
            worst_sub = sub_key

    return {
        "sub_stats": sub_stats,
        "worst_sub": worst_sub,
        "worst_sub_current_mean": round(float(worst_val), 2) if worst_val is not None else None,
    }


def _followswing_false_rate(rows: list[Analysis]) -> float:
    total = 0
    false_count = 0

    for row in rows:
        details = _details_node(row)
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
            false_count += 1

    return false_count / total if total else 0.0


def _followswing_risk_level(false_rate: float) -> str:
    if false_rate >= 0.80:
        return "risk"
    if false_rate >= 0.40:
        return "improve"
    return "ok"


def build_score_stats(cur_rows: list[Analysis], prev_rows: list[Analysis]) -> Dict[str, Any]:
    """Build score_stats from current/previous Analysis windows.

    This service converts raw DB Analysis rows into the statistical state used
    by the report pipeline. It calculates stage means, previous means, deltas,
    direction, sub_stats, weakest sub-metric, and FollowSwing risk fields.
    """
    score_stats: Dict[str, Any] = {}

    for key in SCORE_KEYS:
        if key == "5_FollowSwing_SuccessRate":
            cur_false = _followswing_false_rate(cur_rows)
            prev_false = _followswing_false_rate(prev_rows) if prev_rows else cur_false

            cur_success_rate = 100.0 - (cur_false * 100.0)
            prev_success_rate = 100.0 - (prev_false * 100.0)
            delta = cur_success_rate - prev_success_rate
            direction = "improved" if delta > 1e-9 else ("worsened" if delta < -1e-9 else "flat")

            score_stats[key] = {
                "current_mean": round(cur_success_rate, 2),
                "prev_mean": round(prev_success_rate, 2),
                "delta": round(delta, 2),
                "direction": direction,
                "false_rate_current": round(cur_false, 4),
                "false_rate_prev": round(prev_false, 4),
                "risk_level": _followswing_risk_level(cur_false),
                "success_rate_current": round(cur_success_rate, 2),
                "success_rate_prev": round(prev_success_rate, 2),
            }
            continue

        current_mean = _mean_score(cur_rows, key)
        prev_mean = _mean_score(prev_rows, key) if prev_rows else current_mean
        delta = current_mean - prev_mean
        direction = "improved" if delta > 1e-9 else ("worsened" if delta < -1e-9 else "flat")

        node: Dict[str, Any] = {
            "current_mean": round(current_mean, 2),
            "prev_mean": round(prev_mean, 2),
            "delta": round(delta, 2),
            "direction": direction,
        }

        if key in STAGES:
            node.update(_compute_breakdown_stats(STAGES[key], cur_rows, prev_rows))

        score_stats[key] = node

    return score_stats


def build_score_trend(score_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build total-score trend summary from score_stats."""
    cur_avg = float(score_stats.get("total_score", {}).get("current_mean", 0.0))
    prev_avg = float(score_stats.get("total_score", {}).get("prev_mean", cur_avg))
    delta = round(cur_avg - prev_avg, 2)
    direction = "improved" if delta > 1e-9 else ("worsened" if delta < -1e-9 else "flat")

    return {
        "current_mean_average_score": round(cur_avg, 2),
        "prev_mean_average_score": round(prev_avg, 2),
        "delta_average_score": delta,
        "direction": direction,
    }


def build_score_report_state(
    cur_rows: list[Analysis],
    prev_rows: list[Analysis],
) -> Dict[str, Any]:
    """Build score_json-derived state used by the LLM report route."""
    score_stats = build_score_stats(cur_rows, prev_rows)
    trend = build_score_trend(score_stats)

    return {
        "score_stats": score_stats,
        "trend": trend,
    }