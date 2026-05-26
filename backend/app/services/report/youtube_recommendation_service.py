from __future__ import annotations

import logging
import os
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger("app.llm.youtube_tool")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search").strip()
YOUTUBE_MAX_RESULTS_PER_METRIC = int(os.getenv("YOUTUBE_MAX_RESULTS_PER_METRIC", "1")) # metric당 1개씩
YOUTUBE_SEARCH_DEPTH = os.getenv("YOUTUBE_SEARCH_DEPTH", "basic").strip() or "basic"


STAGE_LABELS: Dict[str, str] = {
    "ready": "준비 자세",
    "rotation": "회전 동작",
    "backswing": "백스윙",
    "impact": "임팩트",
    "followswing": "팔로스윙",
}


METRIC_YOUTUBE_QUERY_MAP: Dict[str, str] = {
    # Ready
    "Arm_Angle": "badminton ready position racket arm elbow angle tutorial",
    "Left_Wrist_Height": "badminton ready position non racket arm wrist height balance tutorial",
    "Stance_Width": "badminton ready stance foot width balance tutorial",
    "Wrist_Height_Ratio": "badminton ready position racket wrist height shoulder level tutorial",

    # Rotation
    "Hip_Level": "badminton overhead clear hip rotation body turn tutorial",
    "Shoulder_Ratio": "badminton overhead clear shoulder rotation body turn tutorial",

    # Backswing
    "Wrist_X_Depth": "badminton backswing racket hand position behind shoulder tutorial",
    "Elbow_Lift": "badminton backswing elbow position elbow lift tutorial",
    "L_Shape_Angle": "badminton backswing L shape arm angle tutorial",

    # Impact
    "Arm_Extension_Angle": "badminton overhead clear impact arm extension straight elbow tutorial",
    "Impact_Wrist_Height_Ratio": "badminton overhead clear impact wrist height contact point tutorial",

    # FollowSwing
    "Performance": "badminton overhead clear follow through swing finish tutorial",
}


def _safe_str(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()
    return host.endswith("youtube.com") or host.endswith("youtu.be")


def _metric_query_text(stage: str, metric: str) -> str:
    stage = _safe_str(stage).lower().strip()
    metric = _safe_str(metric).strip()

    # Impact에도 Wrist_Height_Ratio가 있을 수 있으므로 Ready와 구분한다.
    lookup_key = metric
    if stage == "impact" and metric == "Wrist_Height_Ratio":
        lookup_key = "Impact_Wrist_Height_Ratio"

    mapped = METRIC_YOUTUBE_QUERY_MAP.get(lookup_key)
    if mapped:
        return mapped

    stage_label = STAGE_LABELS.get(stage, stage)
    readable_metric = metric.replace("_", " ")
    return f"badminton {stage_label} {readable_metric} correction tutorial"


def collect_weak_sub_scores(meta: Dict[str, Any], threshold: float = 90.0) -> List[Dict[str, Any]]:
    """Collect sub-metrics whose current_mean is below the threshold.

    Args:
        meta: report meta containing score_stats.
        threshold: sub_score threshold. Default is 90.

    Returns:
        Weak metric list with stage, metric, sub_key, score, direction.
    """
    score_stats = (meta or {}).get("score_stats", {}) or {}

    total_to_stage = {
        "1_Ready_Total": "ready",
        "2_Rotation_Total": "rotation",
        "3_Backswing_Total": "backswing",
        "4_Impact_Total": "impact",
    }

    weak_items: List[Dict[str, Any]] = []

    for total_key, stage in total_to_stage.items():
        node = score_stats.get(total_key, {}) or {}
        if not isinstance(node, dict):
            continue

        direction = _safe_str(node.get("direction") or "flat")
        sub_stats = node.get("sub_stats") or {}
        if not isinstance(sub_stats, dict):
            continue

        for sub_key, sub_node in sub_stats.items():
            if not isinstance(sub_node, dict):
                continue
            try:
                score = float(sub_node.get("current_mean"))
            except Exception:
                continue

            if score >= threshold:
                continue

            metric = _safe_str(sub_key)
            if "." in metric:
                metric = metric.split(".")[-1]

            weak_items.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "sub_key": _safe_str(sub_key),
                    "score": round(score, 2),
                    "direction": direction,
                }
            )

    return weak_items


def build_youtube_query(stage: str, metric: str) -> str:
    """Build a YouTube-oriented search query for Tavily.

    Tavily is a web search API, not the official YouTube Data API.
    Therefore, the query is optimized for web search and combined with
    include_domains=["youtube.com", "youtu.be"] in the search request.
    """
    base_query = _metric_query_text(stage, metric)
    return f"{base_query} badminton coaching youtube"


def _tavily_search_youtube(query: str, max_results: int) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY is not set. youtube recommendation skipped query=%s", query)
        return []

    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": YOUTUBE_SEARCH_DEPTH,
        "max_results": max_results,
        "include_domains": ["youtube.com", "youtu.be"],
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
            response = client.post(TAVILY_SEARCH_URL, json=body)
        if response.status_code >= 400:
            logger.warning(
                "Tavily youtube search failed status=%s query=%s body=%s",
                response.status_code,
                query,
                response.text[:300],
            )
            return []
        data = response.json()
    except Exception as e:
        logger.warning("Tavily youtube search exception query=%s err=%s", query, str(e))
        return []

    results = data.get("results") if isinstance(data, dict) else []
    if not isinstance(results, list):
        return []

    videos: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue
        url = _safe_str(item.get("url"))
        if not url or not _is_youtube_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)

        videos.append(
            {
                "title": _safe_str(item.get("title")) or "YouTube coaching video",
                "url": url,
                "thumbnail": _youtube_thumbnail(url),
                "content": _safe_str(item.get("content")),
                "score": item.get("score"),
            }
        )

    return videos


def recommended_youtube_tool(
    meta: Dict[str, Any],
    threshold: float = 90.0,
    max_results_per_metric: int | None = None,
) -> List[Dict[str, Any]]:
    """Recommend YouTube coaching links for every weak sub-metric.

    Policy:
    - YouTube is not used as core evidence for merge_decider.
    - It is always generated separately as supplementary material.
    - Target items are sub_score < threshold.
    """
    per_metric = max_results_per_metric or YOUTUBE_MAX_RESULTS_PER_METRIC
    weak_items = collect_weak_sub_scores(meta or {}, threshold=threshold)

    recommendations: List[Dict[str, Any]] = []
    for item in weak_items:
        stage = _safe_str(item.get("stage"))
        metric = _safe_str(item.get("metric"))
        query = build_youtube_query(stage, metric)
        logger.info(
            "[YouTube Search] stage=%s metric=%s score=%s query=%s",
            stage,
            metric,
            item.get("score"),
            query,
        )

        videos = _tavily_search_youtube(query=query, max_results=per_metric)

        logger.info(
            "[YouTube Result] stage=%s metric=%s result_count=%d urls=%s",
            stage,
            metric,
            len(videos),
            [v.get("url") for v in videos],
        )

        recommendations.append(
            {
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "metric": metric,
                "sub_key": _safe_str(item.get("sub_key")),
                "score": item.get("score"),
                "direction": _safe_str(item.get("direction")),
                "query": query,
                "videos": videos,
            }
        )

    logger.info(
        "recommended_youtube generated weak_count=%d recommendation_count=%d",
        len(weak_items),
        len(recommendations),
    )
    return recommendations

def _extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if host.endswith("youtu.be"):
        return path.strip("/").split("/")[0]

    if host.endswith("youtube.com"):
        if path == "/watch":
            qs = parse_qs(parsed.query or "")
            return _safe_str((qs.get("v") or [""])[0]).strip()
        if path.startswith("/shorts/"):
            return path.split("/shorts/", 1)[-1].split("/", 1)[0]
        if path.startswith("/embed/"):
            return path.split("/embed/", 1)[-1].split("/", 1)[0]

    return ""

def _youtube_thumbnail(url: str) -> str:
    """Return a standard YouTube thumbnail URL from a YouTube watch/short/youtu.be URL."""
    video_id = _extract_youtube_video_id(url)
    if not video_id:
        return ""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"