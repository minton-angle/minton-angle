from __future__ import annotations

import json
import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from functools import lru_cache

logger_llm = logging.getLogger("app.llm")

# ------------------------------------------------------------------
# Groq Settings
# ------------------------------------------------------------------
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "800"))
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.8"))


# ------------------------------------------------------------------
# RAG (Chroma) Settings
# ------------------------------------------------------------------
COACH_KB_PATH = os.getenv("COACH_KB_PATH", "")  # json or jsonl
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_coach_kb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "coach_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
COACH_RAG_TOPK = int(os.getenv("COACH_RAG_TOPK", "6"))
COACH_RAG_MAX_CHARS = int(os.getenv("COACH_RAG_MAX_CHARS", "450"))


def _safe_str(x: Any) -> str:
    try:
        s = "" if x is None else str(x)
    except Exception:
        s = ""
    return s


def _doc_text(d: Dict[str, Any]) -> str:
    # 검색 성능을 위해 메타+제목+본문을 함께 임베딩
    stage = _safe_str(d.get("stage"))
    metric = _safe_str(d.get("metric"))
    band = _safe_str(d.get("score_band"))
    title = _safe_str(d.get("title"))
    content = _safe_str(d.get("content"))
    return f"[{stage}] [{metric}] [{band}] {title} {content}".strip()


def _read_kb(path: str) -> list[Dict[str, Any]]:
    path = _safe_str(path).strip()
    if not path:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except Exception as e:
        logger_llm.warning("RAG KB read failed path=%s err=%s", path, str(e))
        return []

    if not raw:
        return []

    # jsonl
    if "\n" in raw and not raw.lstrip().startswith("["):
        docs: list[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
        return docs

    # json
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict) and isinstance(obj.get("documents"), list):
            return [x for x in obj.get("documents") if isinstance(x, dict)]
        return []
    except Exception as e:
        logger_llm.warning("RAG KB parse failed path=%s err=%s", path, str(e))
        return []


@lru_cache(maxsize=1)
def _get_chroma():
    """Lazy-load Chroma + embedding model. Returns (collection, embedder) or (None, None)."""
    # Optional deps: keep backend running even if RAG deps are not installed yet.
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger_llm.warning("RAG deps missing (install chromadb sentence-transformers). err=%s", str(e))
        return None, None

    try:
        # ✅ CHROMA_DIR 없으면 자동 생성
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
        except Exception as e:
            logger_llm.warning(
                "RAG chroma dir mkdir failed dir=%s err=%s",
                CHROMA_DIR,
                str(e),
            )
        # NOTE: Newer Chroma versions deprecate `chroma_db_impl` Settings.
        # Use PersistentClient for on-disk persistence.
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_or_create_collection(name=CHROMA_COLLECTION)
        embedder = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        logger_llm.warning("RAG init failed err=%s", str(e))
        return None, None

    # KB가 이미 들어있다면 그대로 사용
    try:
        existing = col.count() if hasattr(col, "count") else 0
    except Exception:
        existing = 0

    if existing and existing > 0:
        return col, embedder

    # KB 로드 후 1회 적재
    docs = _read_kb(COACH_KB_PATH)
    if not docs:
        logger_llm.info("RAG KB empty; skip ingest")
        return col, embedder

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[Dict[str, Any]] = []
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        did = _safe_str(d.get("id") or f"kb_{i}").strip() or f"kb_{i}"
        ids.append(did)
        documents.append(_doc_text(d))
        metadatas.append(
            {
                "stage": _safe_str(d.get("stage")),
                "metric": _safe_str(d.get("metric")),
                "score_band": _safe_str(d.get("score_band")),
                "title": _safe_str(d.get("title")),
                "tags": _safe_str(d.get("tags")),
            }
        )

    try:
        embs = embedder.encode(documents, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embs)
        logger_llm.info("RAG KB ingested count=%d dir=%s", len(ids), CHROMA_DIR)
    except Exception as e:
        logger_llm.warning("RAG KB ingest failed err=%s", str(e))

    return col, embedder


def _score_band_from_mean(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if v < 80:
        return "<80"
    if v < 90:
        return "80-90"
    return ">=90"

# 쿼리 빌더: meta.score_stats의 Total과 worst_sub, risk_level을 기반으로 검색 의도에 맞는 텍스트 쿼리를 생성하여 RAG 검색에 사용
def _build_rag_queries(meta: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Build deterministic queries from meta.score_stats (worst_sub + risk_level)."""
    score_stats = (meta or {}).get("score_stats", {}) or {}

    total_to_stage = {
        "1_Ready_Total": "ready",
        "2_Rotation_Total": "rotation",
        "3_Backswing_Total": "backswing",
        "4_Impact_Total": "impact",
    }

    queries: list[Dict[str, Any]] = []

    for total_key, stage in total_to_stage.items():
        node = score_stats.get(total_key, {}) or {}
        cm = node.get("current_mean")
        worst_sub = node.get("worst_sub")
        direction = _safe_str(node.get("direction") or "flat")

        try:
            cm_f = float(cm)
        except Exception:
            cm_f = None

        # Total < 90: 반드시 원인 구체화 대상
        if cm_f is not None and cm_f < 90 and worst_sub:
            band = _score_band_from_mean(cm_f)
            # worst_sub may come as "Stage.Metric" (e.g., "Impact.Wrist_Height_Ratio")
            # For RAG metric matching, use metric-only (remove stage prefix if present)
            metric_only = _safe_str(worst_sub)
            if "." in metric_only:
                metric_only = metric_only.split(".")[-1]
            text = f"{stage} {total_key} worst_sub={worst_sub} band={band} direction={direction}".strip()
            queries.append({"q": text, "where": {"stage": stage, "metric": metric_only, "score_band": band}})
        # Total >= 90: 강점/실수방지용(선택) — 과도한 검색 방지 위해 1개만 뽑기

    # FollowSwing: risk_level improve/risk면 부상 예방 관찰 포인트 문서 우선
    fs = score_stats.get("5_FollowSwing_SuccessRate", {}) or {}
    risk_level = _safe_str(fs.get("risk_level"))
    if risk_level in ("improve", "risk"):
        text = f"followswing injury_prevention risk_level={risk_level}".strip()
        queries.append({"q": text, "where": {"stage": "followswing", "score_band": risk_level}})

    # # Average_Score 트렌드(선택): 요약 문장 톤 다양성용
    """
    Average_Score을 주석처리 함으로써:
    Total < 90 + worst_sub 기반 쿼리
    FollowSwing risk/improve 기반 쿼리
    만 생성하고,
    where={} 전체 검색은 더 이상 실행되지 않음(어떤 쿼리든지 항상 참조하던 "overall trend direction=..." 텍스트 쿼리 제거)
    """
    # tr = (meta or {}).get("trend", {}) or {}
    # ddir = _safe_str(tr.get("direction") or "flat")
    # text = f"overall trend direction={ddir}".strip()
    # queries.append({"q": text, "where": {}})

    # 중복 제거(텍스트 기준)
    seen = set()
    out = []
    for it in queries:
        key = _safe_str(it.get("q"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _retrieve_coaching(meta: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Retrieve coaching snippets from Chroma and return compact list for prompt injection."""
    col, embedder = _get_chroma()
    if col is None or embedder is None:
        return []

    queries = _build_rag_queries(meta or {})

    def _normalize_where(where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Chroma where clause normalizer.

        Chroma expects where to be expressed with exactly one operator at the top-level.
        This converts simple equality dicts into operator form.

        - {} or invalid -> None
        - {"k": v} -> {"k": {"$eq": v}}
        - {"k1": v1, "k2": v2} -> {"$and": [{"k1": {"$eq": v1}}, {"k2": {"$eq": v2}}]}
        """
        if not isinstance(where, dict) or not where:
            return None

        items = [(str(k), where[k]) for k in where.keys() if k is not None]
        if not items:
            return None

        if len(items) == 1:
            k, v = items[0]
            return {k: {"$eq": v}}

        return {"$and": [{k: {"$eq": v}} for k, v in items]}

    # RAG 쿼리 로그: 검색 의도 파악 및 디버깅용
    logger_llm.info("RAG queries=%s", json.dumps(queries, ensure_ascii=False))

    if not queries:
        return []

    results: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 각 쿼리당 2개씩만, 전체 COACH_RAG_TOPK까지
    per_q = 2

    for q in queries:
        if len(results) >= COACH_RAG_TOPK:
            break

        text = _safe_str(q.get("q"))
        where_raw = q.get("where") if isinstance(q.get("where"), dict) else {}
        where = _normalize_where(where_raw)

        try:
            qemb = embedder.encode([text], normalize_embeddings=True).tolist()
            # 쿼리가 들어오면 검색 시도 로그
            res = col.query(
                query_embeddings=qemb,
                n_results=min(per_q, COACH_RAG_TOPK),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except TypeError:
            # older chroma versions may not support include param
            res = col.query(
                query_embeddings=qemb,
                n_results=min(per_q, COACH_RAG_TOPK),
                where=where,
            )
        except Exception as e:
            logger_llm.warning("RAG query failed q=%s err=%s", text, str(e))
            continue

        ids = (res.get("ids") or [[]])[0] if isinstance(res, dict) else []
        docs = (res.get("documents") or [[]])[0] if isinstance(res, dict) else []
        metas = (res.get("metadatas") or [[]])[0] if isinstance(res, dict) else []

        for i, did in enumerate(ids):
            if len(results) >= COACH_RAG_TOPK:
                break
            sid = _safe_str(did)
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)

            doc = _safe_str(docs[i] if i < len(docs) else "")
            md = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}

            # prompt 폭발 방지: 문서 길이 제한
            if COACH_RAG_MAX_CHARS > 0 and len(doc) > COACH_RAG_MAX_CHARS:
                doc = doc[:COACH_RAG_MAX_CHARS].rstrip() + "…"

            results.append(
                {
                    "id": sid,
                    "stage": _safe_str(md.get("stage")),
                    "metric": _safe_str(md.get("metric")),
                    "score_band": _safe_str(md.get("score_band")),
                    "title": _safe_str(md.get("title")),
                    "content": doc,
                }
            )
        # 각 쿼리 처리 후 누적 결과 로그(쿼리별)
        logger_llm.info(
            "RAG retrieved 누적 개수 count=%d ids=%s",
            len(results),
            [r.get("id") for r in results],
        )

    # 최종 누적 결과 로그
    logger_llm.info(
        "RAG retrieved 최종 누적(주입문서) 개수 count=%d ids=%s",
        len(results),
        [r.get("id") for r in results],
    )

    return results


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 분석 AI 코치입니다.

[절대 규칙]
0) `meta.retrieved_coaching`가 제공되면, 각 섹션의 focus_two는 해당 근거를 우선 사용하여 작성하십시오.
   - retrieved_coaching의 문구를 그대로 길게 복붙하지 말고, 핵심 근거를 요약/재서술하여 자연스럽게 반영하십시오.
   - retrieved_coaching가 비어있는 경우에만 일반 코칭 지식으로 작성하십시오.
1) 수치는 반드시 `meta.score_stats`에 있는 값만 사용하십시오.
    - 사용 가능 키: 1_Ready_Total, 2_Rotation_Total, 3_Backswing_Total, 4_Impact_Total, 5_FollowSwing_SuccessRate, total_score
    - 각 Total 키(1~4)는 추가로 아래 정보를 포함할 수 있습니다:
     • sub_stats: { 세부키: { current_mean, prev_mean, delta, direction } }
     • worst_sub, worst_sub_current_mean
    - 5_FollowSwing_SuccessRate는 성공률 점수(0~100)이며, 추가 필드 false_rate_current/false_rate_prev/risk_level을 함께 제공받을 수 있습니다.
1-1) 팔로스윙 섹션에서 risk_level이 improve 또는 risk인 경우에는 '부상 예방/주의' 관찰 포인트를 반드시 1개 이상 포함하십시오.
   - 단, 의학적 진단/확정 표현 금지(예: "어깨 충돌이다", "부상이다").
   - 허용 톤(관찰/주의): "부담이 커질 수 있어요", "통증이 있으면 강도를 낮출 필요가 있어요", "지속되면 전문가 상담을 고려할 수 있어요".
   - risk_level=ok인 경우에는 부상 위험 언급을 하지 마십시오.
   - 각 키별 사용 가능 값: current_mean, prev_mean, delta, direction
   - 사용 금지: angles(단일 세션 값), raw angle, 임의로 만든 수치/예시 수치
1-2) 각 섹션은 서로 다른 신체/동작 관찰 영역을 다뤄야 합니다.
   - ready(준비): 스탠스, 상체 높이, 팔 위치, 준비 타이밍 중 최소 2개 포함
   - rotation(회전): 골반 회전, 체간 분리, 중심축 유지, 하체-상체 연결 중 최소 2개 포함
   - backswing(백스윙): 팔꿈치 위치, 손목 각도, 라켓 준비 경로 중 최소 2개 포함
   - impact(임팩트): 타점 위치, 라켓 각도, 임팩트 순간 체중 이동 중 최소 2개 포함
   - followswing(팔로스윙): 스윙 마무리 높이, 어깨/팔꿈치 부담 여부, 과회전 여부 중 최소 2개 포함
1-3) Total(요약) 점수가 90 미만인 섹션(ready/rotation/backswing/impact)에서는,
   해당 섹션의 meta.score_stats["<TotalKey>"].worst_sub(가장 낮은 세부 항목)을 반드시 1회 이상 언급하여
   '왜 점수가 흔들릴 수 있는지'를 구체화하십시오.
   - 단, 세부 점수 수치는 sub_stats의 값만 사용하고 임의 추정 금지.
   - Total 점수가 90 이상인 경우에는 worst_sub 언급은 선택(강점 설명에 쓰면 됨)입니다.
2) 각 섹션(ready/rotation/backswing/impact/followswing)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정은 입력의 direction 값을 그대로 따르십시오.
   - improved: delta > 0 (점수 상승)
   - worsened: delta < 0 (점수 하락)
   - flat: delta == 0
4) 문구에는 반드시 "이전 기간 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - summary: string
   - growth: { direction: improved|worsened|flat, delta_average_score: number, message: string }
   - sections: {
       ready: { title, change_one, focus_two },
       rotation: { title, change_one, focus_two },
       backswing: { title, change_one, focus_two },
       impact: { title, change_one, focus_two },
       followswing: { title, change_one, focus_two }
     }
   - today_checklist: string[]
6) change_one에는 아래 3개 값을 반드시 포함하십시오(점수 표기, 소수점 2자리):
   - current_mean, prev_mean, delta
7) focus_two는 2개 항목의 배열로 작성하십시오.
   - 각 항목은 1~2문장으로 구성된 문장형 코치 피드백이어야 합니다.
   - 단순 체크형 표현("~이 유지되는지") 대신,
     왜 중요한지 또는 동작에 어떤 영향을 주는지를 포함한 설명형 문장으로 작성하십시오.
   - 지시형(해라/하세요/줄이세요 등 명령형)은 금지합니다.
   - 같은 문장 구조나 어미를 반복하지 마십시오.
8) today_checklist는 정확히 3개 항목의 배열로 작성하십시오.
9) 각 섹션은 current_mean(점수)에 따라 피드백 목적이 달라야 합니다.
   - current_mean >= 90: "유지/강점 확인" 중심으로 작성 (문제 지적 금지)
   - 80 <= current_mean < 90: "안정화/흔들림 방지" 중심으로 작성
   - current_mean < 80: "개선 필요" 중심으로 작성
10) focus_two의 2개 문장은 current_mean에 따라 다음 성격을 따라야 합니다.
   - current_mean >= 90:
     (a) 강점 유지 포인트 1개
     (b) 실수 방지 체크 포인트 1개
   - 80 <= current_mean < 90:
     (a) 안정화 포인트 1개
     (b) 흔들릴 때 나타나는 징후 1개
   - current_mean < 80:
     (a) 주요 개선 포인트 1개
     (b) 개선이 되면 기대되는 변화 1개
""".strip()


# ------------------------------------------------------------------
# User Prompt
# ------------------------------------------------------------------
def _user_prompt(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]],
    lang: str,
) -> str:
    m = meta or {}

    # LLM이 반드시 써야 하는 값만 제공(angles는 제공하지 않음: 최신 1건 고정/0.1° 앵커링 방지)
    safe_meta = {
        "post_idx": m.get("post_idx"),
        "range": m.get("range"),
        "summary": m.get("summary", {}),
        "trend": m.get("trend", {}),
        "score_stats": m.get("score_stats", {}),
        "insights": m.get("insights", {}),
        # RAG retrieved snippets (optional)
        "retrieved_coaching": m.get("retrieved_coaching", []),
    }

    schema = {
        "summary": "string",
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_average_score": "number",
            "message": "string",
        },
        "sections": {
            "ready": {"title": "준비", "change_one": "string", "focus_two": "string[]"},
            "rotation": {"title": "회전", "change_one": "string", "focus_two": "string[]"},
            "backswing": {"title": "백스윙", "change_one": "string", "focus_two": "string[]"},
            "impact": {"title": "임팩트", "change_one": "string", "focus_two": "string[]"},
            "followswing": {"title": "팔로스윙", "change_one": "string", "focus_two": "string[]"},
        },
        "today_checklist": "string[]",
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
        "다음 입력(meta.score_stats, meta.trend)을 사용해 '기간 비교 기반' 점수 리포트를 생성하세요.\n"
        "중요: angles/단일 세션 값은 사용 금지이며 입력에도 제공되지 않습니다.\n"
        "작성 규칙:\n"
        "1) ready/rotation/backswing/impact/followswing 분석 내용은 서로 달라야 합니다.\n"
        "2) 각 섹션의 change_one에는 current_mean, prev_mean, delta(모두 점수, 소수점 2자리) 3개를 반드시 포함하세요.\n"
        "3) focus_two는 2개 항목의 배열이며, 지시형(해라/하세요) 문장 금지.\n"
        "4) today_checklist는 정확히 3개 항목의 배열로 작성하세요.\n"
        "5) 숫자는 meta.score_stats 값만 사용하세요. (세부 항목은 sub_stats를 사용할 수 있습니다)\n"
        "6) Total이 90 미만인 섹션은 worst_sub(가장 낮은 세부 항목)을 언급하여 관찰 포인트를 구체화하세요.\n\n"
        f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


# ------------------------------------------------------------------
# Normalize
# ------------------------------------------------------------------
def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _normalize_report(report_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report_obj, dict):
        return {}

    report_obj.setdefault("summary", "-")
    report_obj.setdefault(
        "growth",
        {"direction": "flat", "delta_average_score": 0.0, "message": "-"}
    )

    # Backward-compat: if older key exists, map it
    if isinstance(report_obj.get("growth"), dict) and "delta_average_score" not in report_obj["growth"]:
        if "delta_mean_abs_kf_error" in report_obj["growth"]:
            try:
                report_obj["growth"]["delta_average_score"] = float(report_obj["growth"].get("delta_mean_abs_kf_error") or 0.0)
            except Exception:
                report_obj["growth"]["delta_average_score"] = 0.0

    report_obj.setdefault("actions", {})
    for k, title in [
        ("kf1", "백스윙 동작"),
        ("kf2", "임팩트 동작"),
        ("kf3", "팔로스루 동작")
    ]:
        node = report_obj["actions"].setdefault(
            k,
            {"title": title, "problem_one": "-", "fix_two": []}
        )
        node["fix_two"] = _ensure_list(node.get("fix_two"))

    # New score-based sections
    report_obj.setdefault("sections", {})
    for key, title in [
        ("ready", "준비"),
        ("rotation", "회전"),
        ("backswing", "백스윙"),
        ("impact", "임팩트"),
        ("followswing", "팔로스윙"),
    ]:
        node = report_obj["sections"].setdefault(
            key,
            {"title": title, "change_one": "-", "focus_two": []},
        )
        node["focus_two"] = _ensure_list(node.get("focus_two"))

    # Backward-compat: map score sections -> legacy actions(kf1/kf2/kf3) if actions missing
    if not report_obj.get("actions"):
        report_obj["actions"] = {
            "kf1": {
                "title": "백스윙 동작",
                "problem_one": report_obj["sections"]["backswing"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["backswing"].get("focus_two", []),
            },
            "kf2": {
                "title": "임팩트 동작",
                "problem_one": report_obj["sections"]["impact"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["impact"].get("focus_two", []),
            },
            "kf3": {
                "title": "팔로스루 동작",
                "problem_one": report_obj["sections"]["followswing"].get("change_one", "-"),
                "fix_two": report_obj["sections"]["followswing"].get("focus_two", []),
            },
        }

    report_obj.setdefault("today_checklist", [])
    return report_obj


# ------------------------------------------------------------------
# Groq API Call
# ------------------------------------------------------------------
def _call_groq_chat(messages, model: str = GROQ_MODEL) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    timeout = httpx.Timeout(40.0)
    t0 = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)

    logger_llm.info(
        "Groq status=%s time_ms=%.1f",
        r.status_code,
        (time.perf_counter() - t0) * 1000.0,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_report(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = GROQ_MODEL,
) -> Dict[str, Any]:

    # Enrich meta with RAG retrieved coaching snippets (optional)
    if meta is not None and not (meta.get("retrieved_coaching") or []):
        try:
            meta["retrieved_coaching"] = _retrieve_coaching(meta)
            # RAG 검색 결과 로그: 검색 결과 파악 및 디버깅용
            logger_llm.info(
                "RAG injected into meta count=%d",
                len(meta.get("retrieved_coaching") or []),
            )
        except Exception as e:
            logger_llm.warning("RAG retrieve failed err=%s", str(e))
            meta["retrieved_coaching"] = []

    # 최종 prompt 입력 로그(rag on/off는 enrichment 이후 상태 기준)
    try:
        score_stats = (meta or {}).get("score_stats", {})
        logger_llm.info(
            "LLM prompt inputs range=%s score_stats=%s rag=%s",
            (meta or {}).get("range"),
            json.dumps(score_stats, ensure_ascii=False),
            "on" if ((meta or {}).get("retrieved_coaching") or []) else "off",
        )
    except Exception:
        pass

    messages = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": _user_prompt(angles, meta, lang)},
    ]

    raw = _call_groq_chat(messages, model)
    logger_llm.info("LLM raw(head)=%s", raw)

    try:
        report_obj = json.loads(raw)
        report_obj = _normalize_report(report_obj)
    except Exception:
        raise RuntimeError(f"Invalid JSON from LLM: {raw}")

    report_obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    report_obj.setdefault("model", model)

    logger_llm.info("LLM report=%s", json.dumps(report_obj, ensure_ascii=False))
    return report_obj