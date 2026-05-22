from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional


def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def score_band_from_mean(value: Any) -> str:
    try:
        score = float(value)
    except Exception:
        return ""
    if score < 80:
        return "<80"
    if score < 90:
        return "80-90"
    return ">=90"


METRIC_QUERY_MAP: Dict[str, str] = {
    # ready
    "arm_angle": "배드민턴 준비 자세 라켓 잡은 팔 팔꿈치 각도 라켓을 몸 앞쪽에 잡는 방법 ready position racket arm elbow angle",
    "left_wrist_height": "배드민턴 준비 자세 보조 팔 손목 높이 균형 라켓 준비 non racket arm wrist height balance",
    "stance_width": "배드민턴 준비 자세 양발 간격 스탠스 균형 발 위치 ready stance foot width balance",
    "wrist_height_ratio": "배드민턴 준비 자세 라켓 손목 높이 어깨 높이 라켓을 몸 앞쪽에 잡기 wrist height shoulder level ready position",

    # rotation
    "hip_level": "배드민턴 스윙 몸통 회전 골반 회전 체중 이동 하체 상체 연결 hip rotation body turn power transfer",
    "shoulder_ratio": "배드민턴 스윙 어깨 회전 몸통 회전 라켓 준비 shoulder rotation trunk turn overhead stroke",

    # backswing
    "wrist_x_depth": "배드민턴 백스윙 라켓 손 위치 어깨 뒤로 준비 손목 위치 racket hand behind shoulder backswing preparation",
    "elbow_lift": "배드민턴 백스윙 팔꿈치 들기 팔꿈치 위치 손목보다 팔꿈치 높게 racket preparation elbow lift backswing",
    "l_shape_angle": "배드민턴 백스윙 L자 모양 팔 각도 어깨 팔꿈치 손목 라켓 준비 L shape arm angle backswing",

    # impact
    "arm_extension_angle": "배드민턴 임팩트 팔 펴기 팔꿈치 신전 타점 라켓 맞는 순간 arm extension straight elbow contact point",
    "impact_wrist_height_ratio": "배드민턴 임팩트 손목 높이 팔꿈치보다 손목 높게 타점 wrist height above elbow contact point",

    # followswing
    "performance": "배드민턴 팔로스윙 스윙 마무리 라켓 팔 마무리 손목 팔꿈치 위치 follow through swing finish",
}


STAGE_QUERY_MAP: Dict[str, str] = {
    "ready": "준비 동작 준비 자세 라켓 준비 스탠스 ready position",
    "rotation": "스윙 회전 몸통 회전 골반 어깨 회전 rotation body turn",
    "backswing": "백스윙 라켓 준비 팔꿈치 손목 팔 위치 backswing racket preparation",
    "impact": "임팩트 타점 팔 펴기 손목 라켓 헤드 impact contact point",
    "followswing": "팔로스윙 스윙 마무리 팔 이완 부상 예방 follow through",
}


def metric_query_text(stage: str, metric: str) -> str:
    """fallback 검색과 evidence metadata 표시를 위한 metric/stage semantic query를 생성합니다."""
    stage = _safe_str(stage)
    metric = _safe_str(metric).lower()

    stage_metric_key = f"{stage}_{metric}"

    mapped_metric = METRIC_QUERY_MAP.get(stage_metric_key) or METRIC_QUERY_MAP.get(metric)
    mapped_stage = STAGE_QUERY_MAP.get(stage, stage)

    if mapped_metric:
        return f"badminton {mapped_stage} {mapped_metric}"

    readable_metric = metric.replace("_", " ")
    return f"badminton {mapped_stage} {readable_metric}"


def _append_follow_swing_risk_query(
    *,
    queries: List[Dict[str, Any]],
    score_stats: Dict[str, Any],
) -> None:
    fs = score_stats.get("5_FollowSwing_SuccessRate", {}) or {}
    risk_level = _safe_str(fs.get("risk_level"))
    if risk_level not in ("improve", "risk"):
        return

    text = (
        "badminton follow through swing finish racket arm relaxation "
        "shoulder elbow load injury prevention coaching correction"
    ).strip()
    queries.append(
        {
            "q": text,
            "stage": "followswing",
            "metric": "Performance",
            "query_source": "followswing_risk",
            "where": None,
        }
    )


def _build_queries_from_movement_reasoning(
    *,
    movement_reasoning: Dict[str, Any],
    logger: Optional[logging.Logger],
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    raw_focus = movement_reasoning.get("retrieval_focus") or []
    if not isinstance(raw_focus, list):
        return queries

    for item in raw_focus:
        if not isinstance(item, dict):
            continue

        stage = _safe_str(item.get("stage"))
        metric = _safe_str(item.get("metric"))
        query_intent = _safe_str(item.get("query_intent"))

        if not query_intent and stage and metric:
            query_intent = metric_query_text(stage, metric)
        elif not query_intent:
            continue

        # query_intent는 movement_reasoning_node에서 reference를 참고해 생성된 검색 의도입니다.
        # 여기서는 LLM이 만든 intent를 실제 Chroma 검색 문자열로만 정리합니다.
        text = f"badminton {query_intent}".strip()

        queries.append(
            {
                "q": text,
                "stage": stage,
                "metric": metric.lower(),
                "query_source": "movement_reasoning",
                "query_intent": query_intent,
            }
        )

    return queries


def _collect_weak_metric_candidates(
    *,
    weak_metrics: Any,
    score_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if isinstance(weak_metrics, list) and weak_metrics:
        for item in weak_metrics:
            if not isinstance(item, dict):
                continue
            stage = _safe_str(item.get("stage"))
            metric = _safe_str(item.get("metric"))
            band = _safe_str(item.get("score_band")) or score_band_from_mean(item.get("score"))

            if not stage or not metric or not band:
                continue

            candidates.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "score_band": band,
                    "score": item.get("score"),
                    "direction": _safe_str(item.get("direction") or "flat"),
                    "sub_key": _safe_str(item.get("sub_key")),
                }
            )
        return candidates

    total_to_stage = {
        "1_Ready_Total": "ready",
        "2_Rotation_Total": "rotation",
        "3_Backswing_Total": "backswing",
        "4_Impact_Total": "impact",
    }

    for total_key, stage in total_to_stage.items():
        node = score_stats.get(total_key, {}) or {}
        direction = _safe_str(node.get("direction") or "flat")
        sub_stats = node.get("sub_stats") or {}
        if not isinstance(sub_stats, dict) or not sub_stats:
            continue

        for sub_key, sub_node in sub_stats.items():
            if not isinstance(sub_node, dict):
                continue
            try:
                sub_score = float(sub_node.get("current_mean"))
            except Exception:
                continue

            if sub_score >= 90:
                continue

            metric_only = _safe_str(sub_key).lower()
            if "." in metric_only:
                metric_only = metric_only.split(".")[-1]
            metric_only = metric_only.lower()

            candidates.append(
                {
                    "stage": stage,
                    "metric": metric_only,
                    "score_band": score_band_from_mean(sub_score),
                    "score": sub_score,
                    "direction": direction,
                    "sub_key": _safe_str(sub_key),
                }
            )

    return candidates


# fallback 쿼리 생성 하는 함수: movement reasoning에서 명시적으로 retrieval focus가 없는 경우, weak_metrics와 score_stats를 기반으로 쿼리를 생성
def _build_queries_from_weak_metrics(
    *,
    weak_metrics: Any,
    score_stats: Dict[str, Any],
    logger: Optional[logging.Logger],
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    candidates = _collect_weak_metric_candidates(
        weak_metrics=weak_metrics,
        score_stats=score_stats,
    )

    for item in candidates:
        stage = _safe_str(item.get("stage"))
        metric = _safe_str(item.get("metric"))
        band = _safe_str(item.get("score_band"))

        if not stage or not metric or not band:
            continue

        base_query = metric_query_text(stage, metric)

        text = base_query

        queries.append(
            {
                "q": text,
                "stage": stage,
                "metric": metric,
                "score_band": band,
                "sub_key": _safe_str(item.get("sub_key")),
                "query_source": "weak_metrics",
                "where": None,
            }
        )

    if logger is not None:
        try:
            logger.info(
                "[RAG] weak_metrics_fallback count=%d items=%s",
                len(candidates),
                json.dumps(candidates, ensure_ascii=False),
            )
        except Exception:
            pass

    return queries


def _dedupe_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for item in queries:
        key = _safe_str(item.get("q"))
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_rag_queries(
    meta: Optional[Dict[str, Any]] = None,
    *,
    movement_reasoning: Optional[Dict[str, Any]] = None,
    rag_queries: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """코칭 근거 검색을 위한 RAG 쿼리 목록을 생성합니다.

    우선순위:
    1. LangGraph state의 rag_queries: Query Rewrite 결과가 있으면 그대로 사용
    2. LangGraph state의 movement_reasoning.retrieval_focus
    3. DB/meta 입력의 weak_metrics
    4. DB/meta 입력의 score_stats fallback
    """
    meta = meta or {}

    # rewrite 이후 재검색에서는 state의 rag_queries를 그대로 사용합니다.
    # retrieval_node는 항상 rag_queries 형태의 검색 요청 리스트를 state에 유지합니다.
    if isinstance(rag_queries, list) and rag_queries:
        return rag_queries

    score_stats = meta.get("score_stats", {}) or {}
    weak_metrics = meta.get("weak_metrics") or []
    movement_reasoning = movement_reasoning or {}

    queries: List[Dict[str, Any]] = []

    # movement_reasoning의 retrieval_focus를 기반으로 쿼리를 생성
    if isinstance(movement_reasoning, dict):
        queries = _build_queries_from_movement_reasoning(
            movement_reasoning=movement_reasoning,
            logger=logger,
        )
    # movement_reasoning에서 명시적으로 retrieval_focus가 없는 경우, weak_metrics와 score_stats를 기반으로 fallback 쿼리를 생성
    if not queries:
        queries = _build_queries_from_weak_metrics(
            weak_metrics=weak_metrics,
            score_stats=score_stats,
            logger=logger,
        )

    _append_follow_swing_risk_query(queries=queries, score_stats=score_stats)
    output = _dedupe_queries(queries)

    return output