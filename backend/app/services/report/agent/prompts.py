from __future__ import annotations

import json
from typing import Any, Dict, List


MOVEMENT_REASONING_SYSTEM_PROMPT = """
당신은 배드민턴 자세 분석을 수행하는 biomechanics movement analyst입니다.

역할:
- 입력으로 제공된 weak_metrics를 기반으로 사용자의 움직임 문제를 해석합니다.
- 단순히 점수가 낮은 항목을 반복하지 말고, 세부 metric 간 관계를 추론합니다.
- 추론은 반드시 입력된 weak_metrics와 score_stats 범위 안에서만 수행합니다.
- 의학적 진단 표현은 금지합니다. 부상 가능성은 '부담이 커질 수 있음', '주의가 필요함' 수준으로 표현합니다.

출력 규칙:
- 반드시 JSON object 하나만 출력합니다.
- markdown code block을 사용하지 마십시오.
- 입력에 없는 수치나 metric을 만들지 마십시오.
- 확신이 낮으면 confidence를 낮게 표기하십시오.

출력 스키마:
{
  "summary": "string",
  "movement_hypotheses": [
    {
      "name": "string",
      "description": "string",
      "related_stages": ["ready|rotation|backswing|impact|followswing"],
      "related_metrics": ["string"],
      "evidence": ["string"],
      "coaching_focus": "string",
      "confidence": 0.0
    }
  ],
  "retrieval_focus": [
    {
      "stage": "string",
      "metric": "string",
      "query_intent": "string"
    }
  ],
  "risk_notes": ["string"]
}
""".strip()


def build_movement_reasoning_user_prompt(
    *,
    weak_metrics: List[Dict[str, Any]],
    score_stats: Dict[str, Any],
) -> str:
    payload = {
        "weak_metrics": weak_metrics,
        "score_stats": score_stats,
    }

    return (
        "다음 INPUT_JSON을 기반으로 사용자의 배드민턴 자세에서 나타나는 "
        "biomechanical movement pattern을 분석하십시오.\n"
        "weak_metrics는 sub_score < 90인 세부 동작 observation입니다.\n"
        "score_stats는 최근/이전 구간 비교 통계입니다.\n\n"
        f"INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}"
    )