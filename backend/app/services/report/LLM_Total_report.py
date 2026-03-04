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

    stage = _safe_str(d.get("stage"))
    metric = _safe_str(d.get("metric"))
    band = _safe_str(d.get("score_band"))
    title = _safe_str(d.get("title"))
    content = _safe_str(d.get("content"))
    summary = _safe_str(d.get("summary"))

    def _join_list(v: Any) -> str:
        if isinstance(v, list):
            return " / ".join([_safe_str(x) for x in v if _safe_str(x)])
        return _safe_str(v)

    cause = _join_list(d.get("cause"))
    impact = _join_list(d.get("impact"))
    fix = _join_list(d.get("fix"))
    checklist = _join_list(d.get("checklist"))
    drills = _join_list(d.get("drills"))

    # Keep it compact but informative.
    extra = " ".join(
        [
            f"요약:{summary}" if summary else "",
            f"원인:{cause}" if cause else "",
            f"영향:{impact}" if impact else "",
            f"교정:{fix}" if fix else "",
            f"체크:{checklist}" if checklist else "",
            f"개선방안:{drills}" if drills else "",
        ]
    ).strip()

    base = f"[{stage}] [{metric}] [{band}] {title} {content}".strip()
    return f"{base} {extra}".strip()


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
        def _join_list(v: Any) -> str:
            if isinstance(v, list):
                return " / ".join([_safe_str(x) for x in v if _safe_str(x)])
            return _safe_str(v)

        metadatas.append(
            {
                "stage": _safe_str(d.get("stage")),
                "metric": _safe_str(d.get("metric")),
                "score_band": _safe_str(d.get("score_band")),
                "title": _safe_str(d.get("title")),
                "summary": _safe_str(d.get("summary")),
                "cause": _join_list(d.get("cause")),
                "impact": _join_list(d.get("impact")),
                "fix": _join_list(d.get("fix")),
                "checklist": _join_list(d.get("checklist")),
                "drills": _join_list(d.get("drills")),
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

# 쿼리 빌더: meta.score_stats의 sub_stats(세부 점수)와 worst_sub/risk_level을 기반으로 RAG 검색 쿼리 생성
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

        text = f"{stage} metric={metric} band={band} direction={direction}".strip()
        queries.append(
            {
                "q": text,
                "where": {"stage": stage, "metric": metric, "score_band": band},
            }
        )

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
        text = f"followswing injury_prevention risk_level={risk_level}".strip()
        queries.append({"q": text, "where": {"stage": "followswing", "score_band": risk_level}})

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


def _retrieve_coaching(meta: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Retrieve coaching snippets from Chroma and return compact list for prompt injection."""
    col, embedder = _get_chroma()
    if col is None or embedder is None:
        return []

    queries = _build_rag_queries(meta or {})

    # Chroma의 where 필터는 dict 형태로 {field: value} 또는 {"$and": [{field: value}, ...]} 여야 합니다.
    def _normalize_where(where: Dict[str, Any]) -> Optional[Dict[str, Any]]:

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

            # Prefer structured metadata for prompt injection (more "coach-like"),
            # while keeping the embedded `doc` as a fallback.
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
                    "title": inj_title,
                    "content": inj_content,
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