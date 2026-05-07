from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger_query = logging.getLogger("app.llm.query")


STAGE_QUERY_MAP: Dict[str, str] = {
    "ready": "badminton ready position racket preparation stance balance",
    "rotation": "badminton rotation body turn hip shoulder rotation",
    "backswing": "badminton backswing racket preparation elbow wrist arm angle",
    "impact": "badminton impact contact point wrist elbow arm extension",
    "followswing": "badminton follow through swing finish shoulder relaxation",
}


METRIC_QUERY_MAP: Dict[str, str] = {
    "Arm_Angle": "arm elbow racket angle",
    "Left_Wrist_Height": "non racket arm wrist height balance",
    "Stance_Width": "stance foot width balance",
    "Wrist_Height_Ratio": "wrist height shoulder level racket preparation",
    "Hip_Level": "hip rotation body turn power transfer",
    "Shoulder_Ratio": "shoulder rotation trunk separation",
    "Wrist_X_Depth": "racket hand depth backswing preparation",
    "Elbow_Lift": "elbow lift racket preparation",
    "L_Shape_Angle": "L shape arm angle backswing",
    "Arm_Extension_Angle": "arm extension contact point",
    "Impact_Wrist_Height_Ratio": "impact wrist height contact point",
    "Performance": "follow through swing finish",
}


def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def query_write_tool(
    *,
    stage: str,
    metric: str,
    score: Optional[float] = None,
    retrieval_feedback: Optional[Dict[str, Any]] = None,
    previous_query: Optional[str] = None,
) -> Dict[str, Any]:

    stage = _safe_str(stage).lower()
    metric = _safe_str(metric)

    stage_query = STAGE_QUERY_MAP.get(stage, stage)
    metric_query = METRIC_QUERY_MAP.get(metric, metric.replace("_", " "))

    base_query = f"{stage_query} {metric_query}".strip()

    feedback_reason = ""

    if isinstance(retrieval_feedback, dict):
        feedback_reason = _safe_str(
            retrieval_feedback.get("reason")
            or retrieval_feedback.get("feedback")
        ).lower()

    query = base_query

    if "too broad" in feedback_reason:
        query += " badminton coaching correction drill"

    elif "semantic mismatch" in feedback_reason:
        query += " badminton technique posture coaching"

    elif "low relevance" in feedback_reason:
        query += " badminton movement correction training"

    rag_query = query
    web_query = f"{query} youtube tutorial"

    payload = {
        "rag_query": rag_query,
        "web_query": web_query,
        "metadata": {
            "stage": stage,
            "metric": metric,
            "score": score,
            "previous_query": previous_query,
            "retrieval_feedback": retrieval_feedback,
        },
    }

    logger_query.info(
        "[QUERY WRITE] %s",
        json.dumps(payload, ensure_ascii=False),
    )

    return payload