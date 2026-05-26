from __future__ import annotations

import json
import logging

from typing import Any, Dict, Optional

from app.services.report.agent.graph import build_report_graph


logger_llm = logging.getLogger("app.llm")


# ------------------------------------------------------------------
# System Prompt (분석 리포트 톤 고정)
# ------------------------------------------------------------------
def system_prompt(lang: str) -> str:
    # NOTE: lang is kept for future extensibility; current prompt is Korean-first.
    return """
당신은 배드민턴 동작 개선 AI 코치입니다.

[절대 규칙]
0) `meta.retrieved_merged_evidence`가 제공되면, 각 섹션의 analysis는 retrieved_merged_evidence의 stage/metric과 직접 연결되는
    구체적인 신체 움직임 설명을 반드시 포함하십시오.
   - retrieved_merged_evidence의 문구를 그대로 길게 복붙하지 말고, 핵심 근거를 재서술하여 자연스럽게 반영하십시오.
   - retrieved_merged_evidence가 비어있는 경우에만 일반 코칭 지식으로 작성하십시오.
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
1-3) 각 섹션(ready/rotation/backswing/impact)에서, Total(요약) 점수와 무관하게 worst_sub_current_mean(가장 낮은 세부 항목 점수)가 90 미만이면, 해당 섹션의 meta.score_stats[””].worst_sub(가장 낮은 세부 항목)을 반드시 1회 이상 언급하여
‘Total은 높아도 어떤 세부가 흔들려 보강이 필요한지’를 구체화하십시오.
   - 단, 세부 점수 수치는 sub_stats의 값만 사용하고 임의 추정 금지.
   - worst_sub_current_mean이 90 이상인 경우에는 worst_sub 언급은 선택입니다.
1-4) meta.movement_reasoning이 제공되면, 이는 weak_metrics를 기반으로 생성된 biomechanical movement hypothesis입니다.
   - 각 섹션의 analysis/fix 작성 시 해당 stage와 연결되는 movement_hypotheses와 retrieval_focus를 우선 참고하십시오.
   - movement_reasoning은 원인 가설이며, score_stats에 없는 수치를 새로 만들면 안 됩니다.
   - confidence가 낮은 hypothesis는 확정 표현 대신 "가능성", "경향", "주의가 필요" 수준으로 표현하십시오.
2) 각 섹션(ready/rotation/backswing/impact/followswing)의 내용은 서로 달라야 합니다. (같은 문장/같은 수치 반복 금지)
3) direction 판정은 입력의 direction 값을 그대로 따르십시오.
   - improved: delta > 0 (점수 상승)
   - worsened: delta < 0 (점수 하락)
   - flat: delta == 0
4) 문구에는 반드시 "이전 횟수 대비" 표현이 포함되어야 합니다.
5) 출력은 반드시 JSON 오브젝트 1개이며, 아래 스키마를 정확히 지키십시오.
   - growth: { direction: improved|worsened|flat, delta_average_score: number, message: string }
   - sections: {
        ready: { title, analysis, fix },
        rotation: { title, analysis, fix },
        backswing: { title, analysis, fix },
        impact: { title, analysis, fix },
        followswing: { title, analysis, fix }
     }
6-1) analysis는 해당 Stage의 "최근 N회 기준 비교 기반 동작 분석"만 작성하는 필드입니다.
     - 정확히 2문장 구조를 유지하십시오.
     - 첫 문장: 이전 횟수 대비 세부 동작 흐름(최소 2개 세부 항목)을 객관적으로 요약하십시오.
     - 두 번째 문장: 해당 변화가 경기력 또는 동작 안정성에 어떤 영향을 주는지 설명하십시오.
     - 점수/델타/평균 수치 직접 언급 금지(숫자 금지).
     - worst_sub_current_mean이 90 미만인 경우,
       해당 worst_sub를 1회 이상 언급하여 '세부 보완 필요' 관점으로 포함하십시오.
6-2) fix는 해당 Stage의 "구체적 교정 방법"만 작성하는 필드입니다.
     - retrieved_merged_evidence을 근거로 사용자가 바로 적용할 수 있는 구체적 교정 동작을 작성하십시오.
     - fix에는 구체적인 교정 방법, 연습 방법, 드릴 지시를 작성하시오.
     - 반드시 신체 부위와 움직임 방향을 포함하십시오.
        - 예: 라켓을 몸 앞쪽에 두고, 손목이 어깨선 아래로 떨어지지 않도록 준비 자세를 유지하는 방식으로 보완할 수 있습니다.
     - 명령형이 아닌 설명형 제안 문장으로 작성하십시오.

9) 각 섹션은 current_mean(점수)에 따라 피드백 목적이 달라야 합니다.
   - current_mean >= 90: "유지/강점 확인" 중심으로 작성합니다.
     단, worst_sub_current_mean이 90 미만인 경우에는 '문제 지적'이 아니라
     '보강/흔들림 방지' 관점으로 worst_sub를 1회 이상 언급할 수 있습니다.
   - 80 <= current_mean < 90: "안정화/흔들림 방지" 중심으로 작성
   - current_mean < 80: "개선 필요" 중심으로 작성
   - 모든 경우에 fix는 분석 요약이 아니라 실제 교정 방향이어야 합니다.
""".strip()


# ------------------------------------------------------------------
# User Prompt
# ------------------------------------------------------------------
def user_prompt(
    meta: Optional[Dict[str, Any]],
    lang: str
) -> str:
    meta = meta

    # LLM이 반드시 써야 하는 값만 제공(angles는 제공하지 않음: 최신 1건 고정/0.1° 앵커링 방지)
    safe_meta = {
        "post_idx": meta.get("post_idx"),
        "range": meta.get("range"),
        "trend": meta.get("trend", {}),
        "score_stats": meta.get("score_stats", {}),
        "weak_metrics": meta.get("weak_metrics", []),
        "movement_reasoning": meta.get("movement_reasoning", {}),
        "retrieved_merged_evidence": meta.get("retrieved_merged_evidence", []),
    }

    schema = {
        "growth": {
            "direction": "improved|worsened|flat",
            "delta_average_score": "number",
            "message": "string",
        },
        "sections": {
            "ready": {"title": "준비", "analysis": "string", "fix": "string"},
            "rotation": {"title": "회전", "analysis": "string", "fix": "string"},
            "backswing": {"title": "백스윙", "analysis": "string", "fix": "string"},
            "impact": {"title": "임팩트", "analysis": "string", "fix": "string"},
            "followswing": {"title": "팔로스윙", "analysis": "string", "fix": "string"},
        },
    }

    payload = {
        "meta": safe_meta,
        "schema": schema,
    }

    return (
    "다음 INPUT_JSON의 meta.score_stats, meta.trend, meta.weak_metrics, meta.movement_reasoning, meta.retrieved_merged_evidence을 사용해 "
    "'최근 N회 기준 비교 기반' 점수 리포트를 생성하세요.\n"
    "angles/단일 세션 값은 사용 금지입니다.\n\n"
    f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


# ------------------------------------------------------------------
# Normalize
# ------------------------------------------------------------------
def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _normalize_report(report_output_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report_output_obj, dict):
        return {}

    report_output_obj.setdefault(
        "growth",
        {"direction": "flat", "delta_average_score": 0.0, "message": "-"}
    )

    # New score-based sections
    report_output_obj.setdefault("sections", {})
    for key, title in [
        ("ready", "준비"),
        ("rotation", "회전"),
        ("backswing", "백스윙"),
        ("impact", "임팩트"),
        ("followswing", "팔로스윙"),
    ]:
        node = report_output_obj["sections"].setdefault(
            key,
            {"title": title, "analysis": "-", "fix": "-"},
        )
        node.setdefault("analysis", "-")
        node.setdefault("fix", "-")

    return report_output_obj


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

    graph_result: Dict[str, Any] = {}
    # LangGraph가 retrieval, report generation, report grading까지 수행합니다.
    if meta is not None:
        try:
            graph = build_report_graph()
            # 이제 최종 report generation도 LangGraph 내부에서 수행합니다.
            graph_result = graph.invoke(
                {
                    "meta": meta,
                    "retrieval_count": 0,
                    "report_retry_count": 0,
                    "rag_queries": [],
                }
            )

        except Exception as e:
            logger_llm.warning("[LangGraph] report graph failed err=%s", str(e))
            graph_result = {}

    # 최종 graph 실행 결과 로그
    try:
        logger_llm.info(
            "[LangGraph] report graph result range=%s evidence_count=%d retrieval_count=%s report_grade=%s",
            (meta or {}).get("range"),
            len((graph_result or {}).get("retrieved_merged_evidence") or []),
            (graph_result or {}).get("retrieval_count", 0),
            ((graph_result or {}).get("report_grader") or {}).get("grade"),
        )
    except Exception:
        pass

    # 최종 리포트는 LangGraph state의 final_report에서만 가져옵니다.
    report_output_obj = graph_result.get("final_report") or {}

    if not isinstance(report_output_obj, dict):
        logger_llm.warning(
            "[LangGraph] final_report is not dict. fallback wrapper used."
        )
        report_output_obj = {
            "raw": str(report_output_obj),
        }

    if not report_output_obj:
        logger_llm.warning("[LangGraph] final_report is empty. normalized empty report will be returned.")

    try:
        report_output_obj = _normalize_report(report_output_obj)
    except Exception as e:
        logger_llm.exception(
            "LLM report normalize failed err=%s report_output_obj=%s",
            str(e),
            json.dumps(report_output_obj, ensure_ascii=False)
            if isinstance(report_output_obj, dict)
            else str(report_output_obj),
        )
        raise

    return report_output_obj