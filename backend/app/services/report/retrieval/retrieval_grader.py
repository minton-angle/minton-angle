from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from app.services.report.llm.client import call_llm


logger_grader = logging.getLogger("app.llm")

# 문서 관련성 및 커버리지 평가를 위한 시스템 프롬프트
RETRIEVAL_GRADER_SYSTEM_PROMPT = """
당신은 Adaptive RAG Retrieval Evaluator 입니다.

역할:
- 현재 retrieval evidence가 사용자의 movement reasoning을 충분히 설명하는지 평가합니다.
- retrieval quality가 부족하면 retry/rewrite가 필요한지 판단합니다.
- keyword 포함 여부가 아니라 semantic coverage를 평가해야 합니다.

평가 기준:
1. movement_reasoning의 핵심 biomechanical 문제를 retrieval evidence가 실제로 설명하는가?
2. retrieval evidence에 실제 코칭/교정/움직임 근거가 존재하는가?
3. retrieval evidence가 단순 일반론인지, query_intent를 직접적으로 설명하는지 평가하라.
4. retrieval evidence가 부족하다면 어떤 biomechanical concept가 부족한지 반환하라.
5. 입력 payload의 retrieved_docs는 평가 직전에 index 필드가 부여된 문서 목록입니다. 각 문서를 개별 평가하여 최종 리포트 근거로 사용할 문서 index만 filtered_doc_indices에 넣으십시오.
6. filtered_doc_indices에는 movement_reasoning 또는 query_intent와 직접 연결되는 문서만 포함하십시오.
7. 일반론, 다른 stage/metric 문서, 교정 근거가 약한 문서는 filtered_doc_indices에 포함하지 마십시오.
8. 모든 문서가 약하더라도 최종 리포트에 사용할 최소 근거가 있으면 filtered_doc_indices에 포함할 수 있습니다.

출력은 반드시 JSON 객체 하나만 반환하십시오.

JSON schema:
{
  "relevant": true,
  "needs_retry": false,
  "coverage": 0.0,
  "reason": "string",
  "missing_concepts": ["string"],
  "rewrite_guidance": ["string"],
  "filtered_doc_indices": [0, 1],
  "doc_judgements": [
    {
      "index": 0,
      "relevant": true,
      "reason": "string"
    }
  ]
}
""".strip()



def _safe_str(v: Any) -> str:
    try:
        return "" if v is None else str(v)
    except Exception:
        return ""



def _strip_markdown_code_fences(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()

        if inner.lower().startswith("json"):
            inner = inner[4:].strip()

        return inner

    return text



def _extract_json_object(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")

    if start < 0:
        return text

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    return text[start:]


# 검색 문서가 movement_reasoning을 설명 가능하는지 평가하는 함수(relevant)
def grade_retrieval_results(
    *,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    movement_reasoning: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """LLM-based Adaptive RAG retrieval grader."""

    movement_reasoning = movement_reasoning or {}

    indexed_docs = []
    for idx, item in enumerate(retrieved_docs or []):
        if not isinstance(item, dict):
            continue
        indexed_docs.append(
            {
                "index": idx,
                "stage": item.get("stage"),
                "metric": item.get("metric"),
                "source": item.get("source"),
                "content": _safe_str(item.get("content"))[:1800],
            }
        )

    user_payload = {
        "query": query,
        "movement_reasoning": movement_reasoning,
        "retrieved_docs": indexed_docs,
    }

    messages = [
        {
            "role": "system",
            "content": RETRIEVAL_GRADER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]

    try:

        raw = call_llm(messages, model="")

        parsed = json.loads(
            _extract_json_object(
                _strip_markdown_code_fences(raw)
            )
        )
        doc_count = len(retrieved_docs or [])
        filtered_doc_indices = []
        for idx in parsed.get("filtered_doc_indices") or []:
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if 0 <= idx_int < doc_count and idx_int not in filtered_doc_indices:
                filtered_doc_indices.append(idx_int)

        doc_judgements = parsed.get("doc_judgements") or []
        if not isinstance(doc_judgements, list):
            doc_judgements = []

        result = {
            "relevant": bool(parsed.get("relevant", False)),
            "needs_retry": bool(parsed.get("needs_retry", False)),
            "coverage": float(parsed.get("coverage", 0.0)),
            "reason": _safe_str(parsed.get("reason")),
            "missing_concepts": parsed.get("missing_concepts") or [],
            "rewrite_guidance": parsed.get("rewrite_guidance") or [],
            "filtered_doc_indices": filtered_doc_indices,
            "doc_judgements": doc_judgements,
        }

    except Exception as exc:
        logger_grader.warning(
            "LLM retrieval grader failed err=%s",
            str(exc),
        )

        result = {
            "relevant": len(retrieved_docs or []) > 0,
            "needs_retry": len(retrieved_docs or []) == 0,
            "coverage": 0.0,
            "reason": f"fallback grader used: {str(exc)}",
            "missing_concepts": [],
            "rewrite_guidance": [],
            "filtered_doc_indices": list(range(len(retrieved_docs or []))) if retrieved_docs else [],
            "doc_judgements": [],
        }

    logger_grader.info(
        "[LLM_RETRIEVAL_GRADER] %s",
        json.dumps(result, ensure_ascii=False),
    )

    return result