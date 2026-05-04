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
# LLM usage (token counts)
# ------------------------------------------------------------------
# OpenAI-compatible responses may include `usage` like:
# {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
# We keep the last call's usage in-memory so `generate_report()` can attach it.
_LAST_LLM_USAGE: Dict[str, Any] = {}

def _set_last_llm_usage(u: Any) -> None:
    global _LAST_LLM_USAGE
    if isinstance(u, dict):
        _LAST_LLM_USAGE = u
    else:
        _LAST_LLM_USAGE = {}

def _get_last_llm_usage() -> Dict[str, Any]:
    return _LAST_LLM_USAGE if isinstance(_LAST_LLM_USAGE, dict) else {}

# ------------------------------------------------------------------
# LLM Provider Settings (Groq / Hugging Face)
# ------------------------------------------------------------------
# env 파일에서 LLM_PROVIDER 값을 읽어서 사용할 LLM API를 결정함:
# - LLM_PROVIDER=groq (디폴트값, Groq OpenAI-compatible endpoints)
# - LLM_PROVIDER=hf  (Hugging Face OpenAI-compatible endpoints, e.g. Inference Endpoints/TGI)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq (OpenAI-compatible)
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Hugging Face (OpenAI-compatible)
# Examples:
# - HF_BASE_URL=https://<your-endpoint>/v1
# - HF_API_KEY=hf_...  (or provider-specific token)
# - HF_MODEL=<model name> (some endpoints ignore this; keep for compatibility)
HF_BASE_URL = os.getenv("HF_BASE_URL", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "").strip()

# Shared generation params
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "1600")))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.8")))


# Some providers do not support response_format=json_object. Keep it optional.
# - LLM_JSON_MODE=1 to request JSON mode when supported (Groq supports it).
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1").strip() not in ("0", "false", "False")

# Debug: dump raw LLM output on JSON parse error
LLM_DUMP_RAW_ON_ERROR = os.getenv("LLM_DUMP_RAW_ON_ERROR", "0").strip() not in ("0", "false", "False")
LLM_DUMP_RAW_DIR = os.getenv("LLM_DUMP_RAW_DIR", "./snapshots/llm_raw").strip() or "./snapshots/llm_raw"


# ------------------------------------------------------------------
# RAG (Chroma) Settings
# ------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "app/chroma_coach_pdf")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "coach_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
COACH_RAG_TOPK = int(os.getenv("COACH_RAG_TOPK", "6"))
COACH_RAG_MAX_CHARS = int(os.getenv("COACH_RAG_MAX_CHARS", "450"))

# 리포트 생성 시점에는 임베딩 적재를 수행하지 않는다.
# backend/scripts/ingest_rag.py를 먼저 실행해 Chroma를 준비한 뒤, 여기서는 검색만 수행한다.


def _safe_str(x: Any) -> str:
    try:
        s = "" if x is None else str(x)
    except Exception:
        s = ""
    return s






# 검색 전용 LangChain-compatible embedding wrapper
class _E5LangChainEmbeddings:
    """LangChain-compatible embedding wrapper for multilingual-e5 models.

    E5 계열은 문서에는 `passage:`, 쿼리에는 `query:` prefix를 붙이는 사용법이 권장된다.
    LangChain VectorStore가 호출하는 `embed_documents`, `embed_query` 인터페이스에 맞춰 래핑한다.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"passage: {_safe_str(t)}" for t in texts]
        return self.model.encode(passages, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = f"query: {_safe_str(text)}"
        return self.model.encode([query], normalize_embeddings=True).tolist()[0]


@lru_cache(maxsize=1)
def _get_chroma():
    """Lazy-load LangChain Chroma vectorstore for retrieval only.

    임베딩 적재는 backend/scripts/ingest_rag.py에서 수행한다.
    이 함수는 이미 생성된 Chroma collection에 연결하고 검색용 embedding wrapper만 준비한다.

    Returns:
        vectorstore or None
    """
    try:
        try:
            from langchain_chroma import Chroma
        except Exception:
            from langchain_community.vectorstores import Chroma
    except Exception as e:
        logger_llm.warning(
            "LangChain RAG deps missing. install langchain langchain-community langchain-chroma chromadb sentence-transformers. err=%s",
            str(e),
        )
        return None

    try:
        embeddings = _E5LangChainEmbeddings(EMBED_MODEL)
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception as e:
        logger_llm.warning("LangChain RAG init failed err=%s", str(e))
        return None

    try:
        existing = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else 0
    except Exception:
        existing = 0

    if not existing:
        logger_llm.warning(
            "LangChain RAG collection is empty. Run `python backend/scripts/ingest_rag.py --reset` before generating reports. dir=%s collection=%s",
            CHROMA_DIR,
            CHROMA_COLLECTION,
        )
    else:
        logger_llm.info("LangChain RAG collection loaded count=%d dir=%s", existing, CHROMA_DIR)

    return vectorstore


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

# 개발자용 pose metric 이름을 PDF 코칭 문서에서 쓰일 가능성이 높은 자연어 표현으로 변환한다.
# PDF는 `Elbow_Lift`, `Wrist_Height_Ratio` 같은 내부 metric 명칭을 알지 못하므로,
# 검색 쿼리는 사람이 이해하는 배드민턴 자세 표현으로 만들어야 한다.
METRIC_QUERY_MAP: Dict[str, str] = {
    # Ready
    "Arm_Angle": "배드민턴 준비 자세 라켓 잡은 팔 팔꿈치 각도 라켓을 몸 앞쪽에 잡는 방법 ready position racket arm elbow angle",
    "Left_Wrist_Height": "배드민턴 준비 자세 보조 팔 손목 높이 균형 라켓 준비 non racket arm wrist height balance",
    "Stance_Width": "배드민턴 준비 자세 양발 간격 스탠스 균형 발 위치 ready stance foot width balance",
    "Wrist_Height_Ratio": "배드민턴 준비 자세 라켓 손목 높이 어깨 높이 라켓을 몸 앞쪽에 잡기 wrist height shoulder level ready position",

    # Rotation
    "Hip_Level": "배드민턴 스윙 몸통 회전 골반 회전 체중 이동 하체 상체 연결 hip rotation body turn power transfer",
    "Shoulder_Ratio": "배드민턴 스윙 어깨 회전 몸통 회전 라켓 준비 shoulder rotation trunk turn overhead stroke",

    # Backswing
    "Wrist_X_Depth": "배드민턴 백스윙 라켓 손 위치 어깨 뒤로 준비 손목 위치 racket hand behind shoulder backswing preparation",
    "Elbow_Lift": "배드민턴 백스윙 팔꿈치 들기 팔꿈치 위치 손목보다 팔꿈치 높게 racket preparation elbow lift backswing",
    "L_Shape_Angle": "배드민턴 백스윙 L자 모양 팔 각도 어깨 팔꿈치 손목 라켓 준비 L shape arm angle backswing",

    # Impact
    "Arm_Extension_Angle": "배드민턴 임팩트 팔 펴기 팔꿈치 신전 타점 라켓 맞는 순간 arm extension straight elbow contact point",
    "Impact_Wrist_Height_Ratio": "배드민턴 임팩트 손목 높이 팔꿈치보다 손목 높게 타점 wrist height above elbow contact point",

    # FollowSwing
    "Performance": "배드민턴 팔로스윙 스윙 마무리 라켓 팔 마무리 손목 팔꿈치 위치 follow through swing finish",
}

STAGE_QUERY_MAP: Dict[str, str] = {
    "ready": "준비 동작 준비 자세 라켓 준비 스탠스 ready position",
    "rotation": "스윙 회전 몸통 회전 골반 어깨 회전 rotation body turn",
    "backswing": "백스윙 라켓 준비 팔꿈치 손목 팔 위치 backswing racket preparation",
    "impact": "임팩트 타점 팔 펴기 손목 라켓 헤드 impact contact point",
    "followswing": "팔로스윙 스윙 마무리 팔 이완 부상 예방 follow through",
}


def _metric_query_text(stage: str, metric: str) -> str:
    stage = _safe_str(stage)
    metric = _safe_str(metric)

    # Impact에도 Wrist_Height_Ratio가 있으므로 Ready와 구분하기 위해 stage-aware key를 먼저 확인한다.
    stage_metric_key = f"{stage.capitalize()}_{metric}"
    if stage == "impact" and metric == "Wrist_Height_Ratio":
        stage_metric_key = "Impact_Wrist_Height_Ratio"

    mapped_metric = METRIC_QUERY_MAP.get(stage_metric_key) or METRIC_QUERY_MAP.get(metric)
    mapped_stage = STAGE_QUERY_MAP.get(stage, stage)

    if mapped_metric:
        return f"badminton {mapped_stage} {mapped_metric}"

    # fallback: 내부 metric명을 그대로 쓰되 underscore를 공백으로 바꿔 검색 가능성을 높인다.
    readable_metric = metric.replace("_", " ")
    return f"badminton {mapped_stage} {readable_metric}"

# 쿼리 빌더: meta.score_stats의 sub_stats(세부 점수)와 worst_sub/risk_level을 기반으로 RAG 검색 쿼리 생성
def _rewrite_query_with_llm(stage: str, metric: str) -> str:
    prompt = f"""
You are generating a semantic search query for retrieving badminton coaching knowledge.

Rules:
- Output ONLY one short query
- Max 12 words
- No explanation
- No punctuation except spaces
- Use natural coaching language
- Must match wording found in coaching manuals

Input:
stage: {stage}
metric: {metric}

Output:
"""

    messages = [
        {"role": "system", "content": "You generate short search queries."},
        {"role": "user", "content": prompt},
    ]

    try:
        q = _call_llm_chat(messages, model="")
        return q.strip().replace("\n", " ")
    except Exception as e:
        logger_llm.warning("LLM query rewrite failed stage=%s metric=%s err=%s", stage, metric, str(e))
        return ""

def _build_rag_queries(meta: Dict[str, Any]) -> list[Dict[str, Any]]:

    score_stats = (meta or {}).get("score_stats", {}) or {}

    total_to_stage = {
        "1_Ready_Total": "ready",
        "2_Rotation_Total": "rotation",
        "3_Backswing_Total": "backswing",
        "4_Impact_Total": "impact",
    }

    # 1) Collect candidate weak sub-metrics across stages
    candidates: list[Dict[str, Any]] = []

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

            # Only focus weak sub-metrics
            if sub_score >= 90:
                continue

            metric_only = _safe_str(sub_key)
            # sub_key examples: "Ready.Wrist_Height_Ratio" -> "Wrist_Height_Ratio"
            if "." in metric_only:
                metric_only = metric_only.split(".")[-1]

            band = _score_band_from_mean(sub_score)

            candidates.append(
                {
                    "stage": stage,
                    "metric": metric_only,
                    "score_band": band,
                    "score": sub_score,
                    "direction": direction,
                    "sub_key": _safe_str(sub_key),
                }
            )

    # 정책 변경:
    # - sub_score < 90 인 모든 세부 항목을 RAG 대상으로 사용
    # - stage별 개수 제한 제거
    # - 전체 상위 N개 제한 제거

    queries: list[Dict[str, Any]] = []

    for c in candidates:
        stage = _safe_str(c.get("stage"))
        metric = _safe_str(c.get("metric"))
        band = _safe_str(c.get("score_band"))
        direction = _safe_str(c.get("direction") or "flat")

        if not stage or not metric or not band:
            continue

        # 자연어 기반 query 생성 (PDF semantic search용)
        # 내부 metric 이름이 아니라 PDF 코칭 문서에 존재할 법한 자세 표현으로 검색한다.
        base_query = _metric_query_text(stage, metric)

        # LLM-based query rewriting (짧고 자연스러운 검색 쿼리 생성)
        llm_query = _rewrite_query_with_llm(stage, metric)

        if llm_query:
            text = llm_query
        else:
            # fallback (기존 rule 기반)
            text = base_query

        queries.append({
            "q": text,
            # PDF RAG에서는 metadata filter 사용하지 않음 (semantic search 중심)
            "stage": stage,
            "metric": metric,
            "score_band": band,
            "sub_key": _safe_str(c.get("sub_key")),
            "where": None,
        })

    # Debug: 전체 sub<90 항목이 모두 쿼리로 변환되었는지 확인
    try:
        logger_llm.info(
            "RAG all_weak_sub count=%d items=%s",
            len(candidates),
            json.dumps(candidates, ensure_ascii=False),
        )
    except Exception:
        pass

    # FollowSwing: risk_level improve/risk면 부상 예방 관찰 포인트 문서 우선
    fs = score_stats.get("5_FollowSwing_SuccessRate", {}) or {}
    risk_level = _safe_str(fs.get("risk_level"))
    if risk_level in ("improve", "risk"):
        text = (
            "badminton follow through swing finish racket arm relaxation "
            "shoulder elbow load injury prevention coaching correction"
        ).strip()
        queries.append({"q": text, "where": None})

    # 중복 제거(텍스트 기준)
    seen = set()
    out: list[Dict[str, Any]] = []
    for it in queries:
        key = _safe_str(it.get("q"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _retrieve_coaching(meta: Dict[str, Any]) -> list[Dict[str, Any]]: # meta.score_stats의 sub_stats와 worst_sub/risk_level 기반으로 RAG 검색 쿼리 생성 및 Chroma에서 관련 문서 검색
    """Retrieve coaching snippets from Chroma and return compact list for prompt injection."""
    vectorstore = _get_chroma()
    if vectorstore is None:
        return []

    queries = _build_rag_queries(meta or {})


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

        try:
            retrieved_pairs = vectorstore.similarity_search_with_score(
                text,
                k=max(min(per_q * 4, 12), per_q),
            )
        except Exception as e:
            logger_llm.warning("LangChain RAG query failed q=%s err=%s", text, str(e))
            continue

        logger_llm.info(f"RAG query='{text}' results={len(retrieved_pairs)}")

        for doc_obj, distance in retrieved_pairs:
            if len(results) >= COACH_RAG_TOPK:
                break

            md = doc_obj.metadata if isinstance(getattr(doc_obj, "metadata", None), dict) else {}
            sid = _safe_str(md.get("id"))
            if not sid:
                sid = f"{_safe_str(md.get('stage'))}:{_safe_str(md.get('metric'))}:{_safe_str(md.get('score_band'))}:{len(results)}"
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            raw_doc = _safe_str(getattr(doc_obj, "page_content", ""))
            doc = raw_doc

            source_file = _safe_str(md.get("source_file"))
            page = _safe_str(md.get("page"))
            chunk = _safe_str(md.get("chunk"))
            preview = raw_doc.replace("\n", " ").strip()
            # if len(preview) > 220:
            #     preview = preview[:220].rstrip() + "…"

            # prompt 폭발 방지: 문서 길이 제한
            if COACH_RAG_MAX_CHARS > 0 and len(doc) > COACH_RAG_MAX_CHARS:
                doc = doc[:COACH_RAG_MAX_CHARS].rstrip() + "…"

            logger_llm.info(
                "RAG hit query=%s source=%s page=%s chunk=%s distance=%s raw_len=%d injected_len=%d preview=%s",
                text,
                source_file,
                page,
                chunk,
                distance,
                len(raw_doc),
                len(doc),
                preview,
            )

            # LangChain Document metadata를 우선 사용해 prompt 주입용 코칭 스니펫을 구성한다.
            # page_content는 구조화 메타데이터가 비어 있을 때 fallback으로 사용한다.
            inj_title = _safe_str(md.get("title"))
            inj_summary = _safe_str(md.get("summary"))
            inj_cause = _safe_str(md.get("cause"))
            inj_impact = _safe_str(md.get("impact"))
            inj_fix = _safe_str(md.get("fix"))
            inj_check = _safe_str(md.get("checklist"))
            inj_drills = _safe_str(md.get("drills"))

            parts = []
            if inj_summary:
                parts.append(f"요약: {inj_summary}")
            if inj_cause:
                parts.append(f"원인: {inj_cause}")
            if inj_impact:
                parts.append(f"영향: {inj_impact}")
            if inj_fix:
                parts.append(f"교정: {inj_fix}")
            if inj_check:
                parts.append(f"체크: {inj_check}")
            if inj_drills:
                parts.append(f"개선방법: {inj_drills}")

            inj_content = "\n".join(parts).strip() or doc

            results.append(
                {
                    "id": sid,
                    "stage": _safe_str(md.get("stage")),
                    "metric": _safe_str(md.get("metric")),
                    "score_band": _safe_str(md.get("score_band")),
                    "title": inj_title or source_file,
                    "content": inj_content,
                    "distance": distance,
                    "doc_type": _safe_str(md.get("doc_type")),
                    "source_file": source_file,
                    "page": page,
                    "chunk": chunk,
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
    try:
        stage_counts: Dict[str, int] = {}
        for r in results:
            st = _safe_str(r.get("stage"))
            stage_counts[st] = stage_counts.get(st, 0) + 1
        logger_llm.info("RAG injected stage_counts=%s", json.dumps(stage_counts, ensure_ascii=False))
    except Exception:
        pass

    return results


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def _system_prompt(lang: str) -> str:
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 분석 AI 코치입니다.

[절대 규칙]
0) `meta.retrieved_coaching`가 제공되면, 각 섹션의 analysis는 retrieved_coaching의 stage/metric과 직접 연결되는
    구체적인 신체 움직임 설명을 반드시 포함하십시오.
   - retrieved_coaching의 문구를 그대로 길게 복붙하지 말고, 핵심 근거를 재서술하여 자연스럽게 반영하십시오.
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
1-3) 각 섹션(ready/rotation/backswing/impact)에서,
   Total(요약) 점수와 무관하게 worst_sub_current_mean(가장 낮은 세부 항목 점수)가 90 미만이면,
   해당 섹션의 meta.score_stats["<TotalKey>"].worst_sub(가장 낮은 세부 항목)을 반드시 1회 이상 언급하여
   'Total은 높아도 어떤 세부가 흔들려 보강이 필요한지'를 구체화하십시오.
   - 단, 세부 점수 수치는 sub_stats의 값만 사용하고 임의 추정 금지.
   - worst_sub_current_mean이 90 이상인 경우에는 worst_sub 언급은 선택입니다.
2) 각 섹션(ready/rotation/backswing/impact/followswing)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정은 입력의 direction 값을 그대로 따르십시오.
   - improved: delta > 0 (점수 상승)
   - worsened: delta < 0 (점수 하락)
   - flat: delta == 0
4) 문구에는 반드시 "이전 횟수 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - growth: { direction: improved|worsened|flat, delta_average_score: number, message: string }
   - sections: {
       ready: { title, analysis },
       rotation: { title, analysis },
       backswing: { title, analysis },
       impact: { title, analysis },
       followswing: { title, analysis }
     }
6-1) analysis는 해당 Stage의 "최근 N회 기준 비교 기반 분석 리포트"를 작성하는 단일 필드입니다.
     - 정확히 3문장 구조를 유지하십시오.
     - 첫 문장: 이전 횟수 대비 세부 동작 흐름(최소 2개 세부 항목)을 객관적으로 요약하십시오.
     - 두 번째 문장: 해당 변화가 경기력 또는 동작 안정성에 어떤 영향을 주는지 설명하십시오.
     - 세 번째 문장: 개선 또는 유지 관점에서의 제안을 작성하십시오.
     - 점수/델타/평균 수치 직접 언급 금지(숫자 금지).
     - 지시형 문장(해라/하세요) 금지, 보고서형 제안 문장으로 작성하십시오.
     - retrieved_coaching가 있는 경우, 핵심 의미만 요약 반영하십시오. (드릴 세부 묘사 금지)
     - worst_sub_current_mean이 90 미만인 경우,
       해당 worst_sub를 1회 이상 언급하여 '세부 보완 필요' 관점으로 포함하십시오.
9) 각 섹션은 current_mean(점수)에 따라 피드백 목적이 달라야 합니다.
   - current_mean >= 90: "유지/강점 확인" 중심으로 작성합니다.
     단, worst_sub_current_mean이 90 미만인 경우에는 '문제 지적'이 아니라
     '보강/흔들림 방지' 관점으로 worst_sub를 1회 이상 언급할 수 있습니다.
   - 80 <= current_mean < 90: "안정화/흔들림 방지" 중심으로 작성
   - current_mean < 80: "개선 필요" 중심으로 작성
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
        "trend": m.get("trend", {}),
        "score_stats": m.get("score_stats", {}),
        "insights": m.get("insights", {}),
        # RAG retrieved snippets (optional)
        "retrieved_coaching": m.get("retrieved_coaching", []),
    }

    schema = {
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_average_score": "number",
            "message": "string",
        },
        "sections": {
            "ready": {"title": "준비", "analysis": "string"},
            "rotation": {"title": "회전", "analysis": "string"},
            "backswing": {"title": "백스윙", "analysis": "string"},
            "impact": {"title": "임팩트", "analysis": "string"},
            "followswing": {"title": "팔로스윙", "analysis": "string"},
        },
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
        "다음 입력(meta.score_stats, meta.trend)을 사용해 '최근 N회 기준 비교 기반' 점수 리포트를 생성하세요.\n"
        "중요: angles/단일 세션 값은 사용 금지이며 입력에도 제공되지 않습니다.\n\n"
        "[필수 규칙] (반드시 지키세요)\n"
        "1) 출력은 JSON 오브젝트 1개만 반환합니다.\n"
        "2) 각 섹션의 analysis는 반드시 '이전 횟수 대비' 문구를 포함합니다.\n"
        "3) 각 섹션의 analysis는 정확히 3문장입니다. 각 문장은 반드시 마침표(.)로 끝나야 합니다.\n"
        "   - 1문장: 이전 횟수 대비 동작 흐름(최소 2개 관찰 포인트) 요약.\n"
        "   - 2문장: 그 변화가 경기력/안정성에 주는 영향.\n"
        "   - 3문장: 개선 또는 유지 관점의 제안(지시형 금지).\n"
        "4) analysis에는 숫자/점수/퍼센트/소수점을 쓰지 않습니다(숫자 금지).\n"
        "7) Total 점수와 무관하게 worst_sub_current_mean이 90 미만인 섹션은, analysis에 worst_sub 문자열을 그대로 1회 이상 포함합니다.\n\n"
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
            {"title": title, "analysis": "-"},
        )
        node.setdefault("analysis", "-")

    # Backward-compat: map score sections -> legacy actions(kf1/kf2/kf3) if actions missing
    if not report_obj.get("actions"):
        report_obj["actions"] = {
            "kf1": {"title": "백스윙 동작", "problem_one": "-", "fix_two": []},
            "kf2": {"title": "임팩트 동작", "problem_one": "-", "fix_two": []},
            "kf3": {"title": "팔로스루 동작", "problem_one": "-", "fix_two": []},
        }

    return report_obj


# ------------------------------------------------------------------
# LLM API Call (OpenAI-compatible)
# ------------------------------------------------------------------
def _chat_completions_url(base_url: str) -> str:
    """Build a chat-completions URL from a base URL.

    Accepts base_url like:
    - https://api.groq.com/openai/v1
    - https://<hf-endpoint>/v1
    - https://<custom-host>

    Returns: <base_url>/chat/completions (with /v1 preserved if provided)
    """
    b = (base_url or "").rstrip("/")
    if not b:
        return ""
    # If caller provided .../v1 already, we still append /chat/completions
    return f"{b}/chat/completions"


def _call_llm_chat(messages, model: str) -> str:
    """Call the configured provider (Groq or Hugging Face) via OpenAI-compatible chat completions."""

    provider = (LLM_PROVIDER or "groq").strip().lower()

    # Log which provider/model is actually being used
    try:
        effective_model = (
            model
            or (HF_MODEL if provider == "hf" else GROQ_MODEL)
            or "model"
        )
        logger_llm.info(
            "LLM call provider=%s base_url=%s model=%s temperature=%.2f max_tokens=%d",
            provider,
            HF_BASE_URL if provider == "hf" else GROQ_BASE_URL,
            effective_model,
            DEFAULT_TEMPERATURE,
            DEFAULT_MAX_TOKENS,
        )
    except Exception:
        pass

    if provider == "hf":
        if not HF_BASE_URL:
            raise RuntimeError("HF_BASE_URL is not set (e.g. https://<your-hf-endpoint>/v1)")
        if not HF_API_KEY:
            raise RuntimeError("HF_API_KEY (or HF_TOKEN) is not set")

        url = _chat_completions_url(HF_BASE_URL)
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }

        # Some HF OpenAI-compatible endpoints ignore `model` (fixed endpoint model), but it is required by schema.
        chosen_model = model or HF_MODEL or "model"

        body = {
            "model": chosen_model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        timeout = httpx.Timeout(60.0)
        t0 = time.perf_counter()

        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=body)

        logger_llm.info(
            "HF status=%s time_ms=%.1f",
            r.status_code,
            (time.perf_counter() - t0) * 1000.0,
        )

        if r.status_code >= 400:
            raise RuntimeError(f"HF API error {r.status_code}: {r.text}")
        data = r.json()
        _set_last_llm_usage(data.get("usage"))
        try:
            if data.get("usage"):
                logger_llm.info("HF usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
        except Exception:
            pass
        return data["choices"][0]["message"]["content"]

    # default: groq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = _chat_completions_url(GROQ_BASE_URL)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    # Groq supports JSON mode; keep optional.
    if LLM_JSON_MODE:
        body["response_format"] = {"type": "json_object"}

    timeout = httpx.Timeout(40.0)
    t0 = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)

        # If provider rejects response_format, retry once without it.
        if r.status_code == 400 and LLM_JSON_MODE and "response_format" in body:
            try:
                txt = r.text or ""
            except Exception:
                txt = ""
            if "response_format" in txt or "json_object" in txt or "response format" in txt.lower():
                body.pop("response_format", None)
                r = client.post(url, headers=headers, json=body)

    logger_llm.info(
        "Groq status=%s time_ms=%.1f",
        r.status_code,
        (time.perf_counter() - t0) * 1000.0,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")
    data = r.json()
    _set_last_llm_usage(data.get("usage"))
    try:
        if data.get("usage"):
            logger_llm.info("Groq usage=%s", json.dumps(data.get("usage"), ensure_ascii=False))
    except Exception:
        pass
    return data["choices"][0]["message"]["content"]


def _strip_markdown_code_fences(s: str) -> str:
    """Remove surrounding Markdown code fences if present (```json ... ```)."""
    s = (s or "").strip()
    if not s:
        return s

    # Handle inline fenced blocks like: ```json { ... }``` or ``` { ... }```
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].strip()
        # Drop optional language tag at the beginning (e.g. json)
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner

    # Handle multi-line fences:
    # ```json\n{...}\n```
    # ```\n{...}\n```
    if s.startswith("```"):
        parts = s.splitlines()
        if parts:
            first = parts[0].strip()
            # If the first line contains JSON right after ```json, keep the remainder.
            # Example: ```json {"a":1}
            if first.startswith("```") and len(first) > 3:
                rest = first[3:].strip()
                # Remove optional language tag
                if rest.lower().startswith("json"):
                    rest = rest[4:].strip()
                if rest:
                    parts = [rest] + parts[1:]
                else:
                    parts = parts[1:]
            else:
                # Plain ``` on first line
                parts = parts[1:]

        s = "\n".join(parts).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        return s

    return s


def _extract_first_json_object(s: str) -> str:
    """Best-effort extraction of the first top-level JSON object from text."""
    s = (s or "").strip()
    if not s:
        return s

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start < 0:
        return s

    # Bracket matching to find the end of the first object
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()

    # If we couldn't match, return from first '{' onward
    return s[start:].strip()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_report(
    angles: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    lang: str = "ko",
    model: str = "",
    system_prompt_override: Optional[str] = None,
    user_prompt_override: Optional[str] = None,
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

    # 오버라이드 허용: 디버깅/실험용으로 system/user prompt를 완전히 교체할 수 있도록 허용
    system_prompt = system_prompt_override if system_prompt_override is not None else _system_prompt(lang)
    user_prompt = user_prompt_override if user_prompt_override is not None else _user_prompt(angles, meta, lang)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = _call_llm_chat(messages, model)
    # Attach token usage (if provider returns it)
    usage = _get_last_llm_usage()
    logger_llm.info("LLM raw(head)=%s", raw)

    raw_clean = _strip_markdown_code_fences(raw)
    raw_clean = _extract_first_json_object(raw_clean)

    try:
        report_obj = json.loads(raw_clean)
        report_obj = _normalize_report(report_obj)
        if usage:
            report_obj["usage"] = usage
    except Exception as e:
        # Debug aid: dump the full raw output to a file when parsing fails.
        if LLM_DUMP_RAW_ON_ERROR:
            try:
                os.makedirs(LLM_DUMP_RAW_DIR, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                safe_model = (model or (HF_MODEL if LLM_PROVIDER == "hf" else GROQ_MODEL) or "model").replace("/", "__")
                out_path = os.path.join(LLM_DUMP_RAW_DIR, f"raw_{safe_model}_{ts}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(raw)
                logger_llm.error("LLM raw dump saved: %s", out_path)
            except Exception as dump_err:
                logger_llm.error("LLM raw dump failed err=%s", str(dump_err))

        # Keep exception small but informative
        raise RuntimeError(f"Invalid JSON from LLM: {raw[:500]}") from e

    report_obj.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    report_obj.setdefault("model", model or (HF_MODEL if LLM_PROVIDER == "hf" else GROQ_MODEL))
    report_obj.setdefault("provider", LLM_PROVIDER)

    # Final confirmation log (after normalization)
    try:
        logger_llm.info(
            "LLM report finalized provider=%s model=%s",
            LLM_PROVIDER,
            report_obj.get("model"),
        )
    except Exception:
        pass

    logger_llm.info("LLM report=%s", json.dumps(report_obj, ensure_ascii=False))
    return report_obj