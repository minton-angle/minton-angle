"""ModelTest.py

A-방안(추천):
1) (선택) API로 meta를 생성해서 파일로 고정(snapshots/meta_*.json)
2) 고정된 meta 파일을 기준으로 Groq 모델 리스트만 바꿔가며 동일 프롬프트로 비교
3) 규칙 위반(스키마/문구/3문장/숫자금지/checklist 3개 등)과 레이턴시를 리포트

실행 예시(프로젝트 루트 기준):
  PYTHONPATH=backend python backend/app/test/TotalReport/ModelTest.py \
    --meta snapshots/meta_sample.json \
    --models llama-3.1-8b-instant mixtral-8x7b-32768 \
    --repeat 3

(meta 자동 생성/저장 - API가 meta를 함께 반환하는 경우):
  PYTHONPATH=backend python backend/app/test/TotalReport/ModelTest.py \
    --fetch-meta --api-base http://localhost:8000 --post-idx user_001 --range 7d \
    --out snapshots/meta_user_001_7d.json

주의:
- 현재 /api/report/post/{post_idx} 응답이 meta를 포함하지 않으면, --fetch-meta는 응답 전체를 저장하고 경고합니다.
  (가장 깔끔한 방법은 reportRouters.py에서 debug 플래그로 meta를 같이 반환하도록 하는 것입니다.)
- 모델 비교의 공정성을 위해 RAG 주입(retrieved_coaching)을 "한 번만" 계산해서 meta에 고정하는 옵션(--freeze-rag)을 제공합니다.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import httpx

# NOTE: LLM_Total_report는 import 시점에 CHROMA_DIR/COACH_KB_PATH 등을 읽습니다.
# 따라서 .env 로딩을 먼저 하고, 그 다음에 모듈을 import 해야 경로가 올바르게 설정됩니다.
_llm_mod = None

def get_llm_mod():
    global _llm_mod
    if _llm_mod is None:
        from app.services.report import LLM_Total_report as _m
        _llm_mod = _m
    return _llm_mod


# -----------------------------
# 기본 설정
# -----------------------------

# 여기서 모델 관리를 하지만, CLI에서 --models를 주면 DEFAULT_MODELS는 무시됨ㅇㅇ
DEFAULT_MODELS = [
    # "llama-3.1-8b-instant",
    # "llama-3.1-70b-versatile",
    # "mixtral-8x7b-32768",
    # "Qwen/Qwen2.5-7B-Instruct"
]

SECTION_KEYS = ["ready", "rotation", "backswing", "impact", "followswing"]

REQUIRED_TOP_KEYS = ["growth", "sections"]

# 자연스러운 시스템 프롬프트 빌더 (natural mode)
def build_natural_system_prompt() -> str:
    return """
당신은 배드민턴 동작 분석 코치입니다.

- 최근 N회 기준 비교 분석을 자연스럽게 작성하십시오.
- 각 섹션은 2~4문장 허용합니다.
- 수치 직접 언급은 피하고 흐름 중심으로 설명하십시오.
- retrieved_coaching이 있으면 핵심 의미만 자연스럽게 반영하십시오.
- JSON 스키마는 반드시 유지하십시오.
""".strip()


# -----------------------------
# 유틸/검증
# -----------------------------

def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _has_digits(s: str) -> bool:
    return any(ch.isdigit() for ch in (s or ""))


def _contains_prev_phrase(s: str) -> bool:
    s = (s or "")
    # New wording (count-based comparison)
    if "이전 횟수 대비" in s:
        return True
    # Backward-compat (older prompt)
    if "이전 기간 대비" in s:
        return True
    return False


def _count_sentences_ko(s: str) -> int:
    """정확한 문장 분리는 어렵기 때문에, 종결 패턴 기반 휴리스틱을 사용합니다.

    시스템 프롬프트는 '정확히 3문장'을 요구하지만, LLM 출력의 종결부호가 다양할 수 있어
    너무 엄격하게 잡으면 false negative가 생깁니다.

    - 기본은 '.', '다.', '요.' 종결을 카운트
    - 문장 끝(개행/문장끝) 종결도 일부 반영
    """
    s = (s or "").strip()
    if not s:
        return 0

    # 1) 마침표 기준(영문/숫자 포함 가능)
    dot_cnt = s.count(".")

    # 2) 한국어 종결 패턴 기준: 마침표 유무 모두 카운트
    # - 예: "...입니다." / "...입니다" / "...다." / "...다" / "...요." / "...요"
    # - 너무 길게 끊기는 것을 방지하기 위해 흔한 종결어미만 포함
    ending_hits = re.findall(r"(습니다\.?|니다\.?|어요\.?|예요\.?|요\.?|다\.?)(?=\s|$)", s)
    end_cnt = len(ending_hits)

    # 3) 휴리스틱 결합:
    # - 마침표가 충분히 있으면 dot_cnt를 우선
    # - 그렇지 않으면 종결어미 카운트를 사용
    if dot_cnt >= 3:
        return dot_cnt
    if end_cnt >= 1:
        return end_cnt

    # 4) 마지막 보정: 문장 끝이면 1로 간주
    return 1




def validate_report_obj(report: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if not isinstance(report, dict):
        return False, ["report_not_dict"]

    # top keys
    for k in REQUIRED_TOP_KEYS:
        if k not in report:
            issues.append(f"missing_top_key:{k}")

    # growth
    growth = report.get("growth")
    if not isinstance(growth, dict):
        issues.append("growth_not_dict")
    else:
        if growth.get("direction") not in ("improved", "worsened", "flat"):
            issues.append("growth_direction_invalid")
        if "delta_average_score" not in growth:
            issues.append("growth_missing_delta_average_score")

    # sections
    sections = report.get("sections")
    if not isinstance(sections, dict):
        issues.append("sections_not_dict")
        return (len(issues) == 0), issues

    # worst_sub 언급 규칙(ready/rotation/backswing/impact)
    score_stats = (meta or {}).get("score_stats", {}) or {}
    total_map = {
        "1_Ready_Total": "ready",
        "2_Rotation_Total": "rotation",
        "3_Backswing_Total": "backswing",
        "4_Impact_Total": "impact",
    }

    for sk in SECTION_KEYS:
        node = sections.get(sk)
        if not isinstance(node, dict):
            issues.append(f"section_not_dict:{sk}")
            continue

        analysis = _safe_str(node.get("analysis"))

        # 공통 규칙: "이전 횟수 대비"(또는 이전 기간 대비) 포함
        if not _contains_prev_phrase(analysis):
            issues.append(f"no_prev_phrase:{sk}")

        # 공통 규칙: analysis 숫자 금지
        if _has_digits(analysis):
            issues.append(f"digit_in_analysis:{sk}")

        # 공통 규칙: 3문장(휴리스틱)
        # - 정확히 3문장을 강제하면 false negative 가능성이 있어, 우선 3 이상/미만으로 분리해 리포트합니다.
        sent_cnt = _count_sentences_ko(analysis)
        if sent_cnt < 3:
            issues.append(f"analysis_sentence_lt_3:{sk}")
        elif sent_cnt > 3:
            issues.append(f"analysis_sentence_gt_3:{sk}")

        # worst_sub 언급(ready/rotation/backswing/impact)
        for total_key, mapped_sk in total_map.items():
            if mapped_sk != sk:
                continue
            st_node = score_stats.get(total_key, {}) or {}
            ws = st_node.get("worst_sub")
            wv = st_node.get("worst_sub_current_mean")
            try:
                wv_f = float(wv) if wv is not None else None
            except Exception:
                wv_f = None

            if wv_f is not None and wv_f < 90 and ws:
                if _safe_str(ws) not in analysis:
                    issues.append(f"missing_worst_sub_mention:{sk}")

    # followswing risk_level 규칙 (improve/risk면 주의/예방 관찰 포인트 포함)
    fs = score_stats.get("5_FollowSwing_SuccessRate", {}) or {}
    risk_level = _safe_str(fs.get("risk_level"))
    if risk_level in ("improve", "risk"):
        fw = sections.get("followswing") if isinstance(sections.get("followswing"), dict) else {}
        analysis = _safe_str((fw or {}).get("analysis"))
        if not any(k in analysis for k in ["부상", "부담", "통증", "주의", "예방"]):
            issues.append("followswing_missing_injury_caution_hint")

    ok = (len(issues) == 0)
    return ok, issues


@dataclass
class RunResult:
    ok: bool
    latency_ms: float
    issues: List[str]


# -----------------------------
# A-방안: API로 meta 생성/저장
# -----------------------------

def fetch_meta_from_api(api_base: str, post_idx: str, range_: str, lang: str) -> Dict[str, Any]:
    """API에서 meta를 가져옵니다.

    - 이상적인 경우: 응답에 {"meta": {...}} 또는 {"data": {"meta": {...}}} 형태로 포함
    - 현재 구현이 meta를 반환하지 않는다면: 응답 전체를 반환(저장용)

    가장 권장되는 방식은 reportRouters.py에 debug 플래그로 meta를 포함해 응답하도록 하는 것입니다.
    """
    url = f"{api_base.rstrip('/')}/api/report/post/{post_idx}"
    # snapshot_only=1 : LLM 호출 없이 meta만 반환(라우터에서 지원)
    params = {"lang": lang, "range": range_, "snapshot_only": "1"}

    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, params=params)

    r.raise_for_status()
    data = r.json()

    # snapshot_only 모드: {"meta": {...}} 형태
    if isinstance(data, dict) and isinstance(data.get("meta"), dict):
        return data["meta"]
    if isinstance(data, dict) and isinstance(data.get("data"), dict) and isinstance(data["data"].get("meta"), dict):
        return data["data"]["meta"]

    # meta가 없으면 전체 응답을 그대로 반환(저장/디버그)
    return {"__warning": "API response did not include meta; saved full response.", "response": data}


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# Helper: Save parsed LLM result (always, regardless of ok) into snapshots/llm_results.
def save_ok_report(
    save_dir: str,
    model: str,
    run_idx: int,
    report: Dict[str, Any],
    meta: Dict[str, Any],
    ok: bool,
    issues: List[str],
) -> str:
    """Save parsed LLM result (always) into snapshots/llm_results."""
    os.makedirs(save_dir, exist_ok=True)

    safe_model = (model or "model").replace("/", "__")
    post_idx = _safe_str(meta.get("post_idx") or "unknown")
    ts = time.strftime("%Y%m%dT%H%M%S")

    mode = getattr(run_benchmark, "_prompt_mode", "strict")
    filename = f"mode{mode}_{safe_model}__{post_idx}__run{run_idx}__{ts}.json"
    path = os.path.join(save_dir, filename)

    payload = {
        "mode": mode,
        "model": model,
        "post_idx": post_idx,
        "run": run_idx,
        "ok": ok,
        "issues": issues,
        "rag_on": bool((meta or {}).get("retrieved_coaching") or []),
        "retrieved_coaching_n": int(len((meta or {}).get("retrieved_coaching") or [])),
        "report": report,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


# -----------------------------
# 모델 비교(고정 meta)
# -----------------------------

# More realistic weighting:
# - critical: deployment-blocking failures
# - major: requirement violations affecting report logic
# - minor: stylistic/format issues (light penalty)
SEVERITY_PENALTY = {
    "critical": 50,   # schema/API/JSON failure → almost unusable
    "major": 10,      # requirement violation but structurally valid
    "minor": 2,       # style/format issue only
}

# -----------------------------
# Issue→severity classification and run scoring helpers
# -----------------------------
def _issue_severity(issue: str) -> str:
    """Map an issue key to a severity bucket for scoring."""
    issue = (issue or "").strip()

    # Critical: schema/structure failures (deployment-blocking)
    if (
        issue == "report_not_dict"
        or issue.startswith("missing_top_key:")
        or issue in ("growth_not_dict", "sections_not_dict")
        or issue.startswith("section_not_dict:")
    ):
        return "critical"

    # Major: requirement/logic violations (still parseable JSON)
    if (
        issue.startswith("growth_")
        or issue.startswith("missing_worst_sub_mention:")
        or issue.startswith("followswing_missing_injury_caution_hint")
        or issue.startswith("no_prev_phrase:")
    ):
        return "major"

    # Minor: style/format violations
    if (
        issue.startswith("analysis_sentence_lt_3:")
        or issue.startswith("analysis_sentence_gt_3:")
        or issue.startswith("digit_in_analysis:")
    ):
        return "minor"

    # Default to major (conservative)
    return "major"


def _score_from_issues(issues: List[str], has_exception: bool) -> int:
    """Compute a 0~100 score from rule violations. Exceptions force a score of 0."""
    if has_exception:
        return 0
    score = 100
    for iss in (issues or []):
        sev = _issue_severity(iss)
        score -= int(SEVERITY_PENALTY.get(sev, 10))
    if score < 0:
        score = 0
    return int(score)

# RAG 주입은 모델마다 다르게 작용할 수 있어, 모델 비교의 공정성을 위해 "한 번만" 계산해서 meta에 고정하는 옵션
def ensure_frozen_rag(meta: Dict[str, Any]) -> Dict[str, Any]:
    """모델 비교의 공정성을 위해 retrieved_coaching을 '한 번만' 계산해 meta에 고정합니다."""
    m = copy.deepcopy(meta)
    if isinstance(m, dict) and not (m.get("retrieved_coaching") or []):
        try:
            m["retrieved_coaching"] = get_llm_mod()._retrieve_coaching(m)
        except Exception:
            m["retrieved_coaching"] = []
    return m


def run_benchmark(meta: Dict[str, Any], models: List[str], repeat: int, lang: str) -> None:
    print("\n============================")
    print("LLM Model Benchmark (fixed meta)")
    print("============================")
    print("models =", models)
    print("repeat =", repeat)
    print("meta.post_idx =", meta.get("post_idx"))
    print("meta.range =", meta.get("range"))
    print("rag =", "on" if (meta.get("retrieved_coaching") or []) else "off")
    print("mode =", getattr(run_benchmark, "_prompt_mode", "strict"))
    print("============================\n")

    all_runs: List[Dict[str, Any]] = []

    for model in models:
        latencies: List[float] = []
        ok_count = 0
        issue_counter: Dict[str, int] = {}
        exc_counter: Dict[str, int] = {}

        for i in range(repeat):
            try:
                # generate_report가 meta를 mutate할 수 있으니 매번 deep copy
                m = copy.deepcopy(meta)
                t0 = time.perf_counter()
                # natural 모드면 시스템 프롬프트 오버라이드 적용
                if getattr(run_benchmark, "_prompt_mode", "strict") == "natural":
                    system_override = build_natural_system_prompt()
                else:
                    system_override = None

                report = get_llm_mod().generate_report(
                    angles={},
                    meta=m,
                    lang=lang,
                    model=model,
                    system_prompt_override=system_override,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                # normalize는 generate_report 내부에서 수행됨
                ok, issues = validate_report_obj(report, meta)
                if ok:
                    ok_count += 1
                else:
                    for iss in issues:
                        issue_counter[iss] = issue_counter.get(iss, 0) + 1

                # --- Always save parsed LLM result ---
                save_dir = "./snapshots/llm_results"
                path = save_ok_report(save_dir, model, i + 1, report, meta, ok, issues)
                print(f"[SAVE] {model} run={i+1} ok={ok} -> {path}")

                # token usage (if provided by provider)
                usage = report.get("usage") if isinstance(report, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = usage.get("total_tokens")
                # tokens/sec = 모델이 1초에 생성한 토큰 수 
                latency_s = dt_ms / 1000.0 if dt_ms else 0.0
                tokens_per_second = (completion_tokens / latency_s) if latency_s > 0 else 0.0
                try:
                    total_tokens = int(total_tokens) if total_tokens is not None else (prompt_tokens + completion_tokens)
                except Exception:
                    total_tokens = prompt_tokens + completion_tokens

                run_score = _score_from_issues(issues, has_exception=False)
                all_runs.append(
                    {
                        "model": model,
                        "run": i + 1,
                        "ok": ok,
                        "score": run_score,
                        "latency_ms": dt_ms,
                        "n_issues": len(issues),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": tokens_per_second,
                        "rag_on": bool((meta or {}).get("retrieved_coaching") or []),
                        "retrieved_coaching_n": int(len((meta or {}).get("retrieved_coaching") or [])),
                        "exception": "",
                    }
                )

            except Exception as e:
                msg = _safe_str(e)
                msg_one = re.sub(r"\s+", " ", msg).strip()
                if len(msg_one) > 140:
                    msg_one = msg_one[:140].rstrip() + "…"

                k = f"exception:{type(e).__name__}:{msg_one}"
                exc_counter[k] = exc_counter.get(k, 0) + 1

                global_seq = i + 1
                all_runs.append(
                    {
                        "model": model,
                        "run": i + 1,
                        "ok": False,
                        "score": 0,
                        "latency_ms": float("nan"),
                        "n_issues": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "tokens_per_second": 0.0,
                        "rag_on": bool((meta or {}).get("retrieved_coaching") or []),
                        "retrieved_coaching_n": int(len((meta or {}).get("retrieved_coaching") or [])),
                        "exception": msg_one,
                    }
                )

        avg = statistics.mean(latencies) if latencies else float("nan")
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else float("nan")
        mn = min(latencies) if latencies else float("nan")
        mx = max(latencies) if latencies else float("nan")

        top_issues = sorted(issue_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        top_excs = sorted(exc_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

        print(f"== Model: {model} ==")
        print(f"latency_ms avg={avg:.1f} p95={p95:.1f} min={mn:.1f} max={mx:.1f}")
        print(f"ok_rate = {ok_count}/{repeat}")
        if top_excs:
            print("exceptions:")
            for k, v in top_excs:
                # k already includes a short reason snippet
                print(f"  - {k}: {v}")
        print("top_issues:")
        if not top_issues:
            print("  - (none)")
        else:
            for k, v in top_issues:
                print(f"  - {k}: {v}")
        print()

    # -----------------------------
    # CSV export (model-level aggregates)
    # -----------------------------
    csv_out = os.getenv("BENCH_CSV_OUT")  # optional env override
    if hasattr(run_benchmark, "_csv_out_override"):
        csv_out = getattr(run_benchmark, "_csv_out_override") or csv_out

    if csv_out:
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for model in models:
            runs = [r for r in all_runs if r.get("model") == model]
            lat = [float(r["latency_ms"]) for r in runs if isinstance(r.get("latency_ms"), (int, float)) and not math.isnan(float(r["latency_ms"]))]

            scores = [int(r["score"]) for r in runs if isinstance(r.get("score"), int)]
            ok_cnt = sum(1 for r in runs if r.get("ok") is True)

            rns = [int(r.get("retrieved_coaching_n") or 0) for r in runs]
            rag_on_cnt = sum(1 for r in runs if r.get("rag_on") is True)
            avg_retrieved_coaching_n = statistics.mean(rns) if rns else float("nan")

            # token usage aggregates
            pts = [int(r.get("prompt_tokens") or 0) for r in runs if isinstance(r.get("prompt_tokens"), int) and int(r.get("prompt_tokens") or 0) > 0]
            cts = [int(r.get("completion_tokens") or 0) for r in runs if isinstance(r.get("completion_tokens"), int) and int(r.get("completion_tokens") or 0) > 0]
            tts = [int(r.get("total_tokens") or 0) for r in runs if isinstance(r.get("total_tokens"), int) and int(r.get("total_tokens") or 0) > 0]
            # 초당 토큰수 정의: 모델이 답변을 내뱉는 속도(출력 토큰 처리량)
            tps = [
                float(r.get("tokens_per_second") or 0.0)
                for r in runs
                if isinstance(r.get("tokens_per_second"), (int, float)) and float(r.get("tokens_per_second") or 0.0) > 0
            ]

            avg_latency = statistics.mean(lat) if lat else float("nan")
            p95_latency = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] if lat else float("nan")
            avg_score = statistics.mean(scores) if scores else float("nan")
            avg_prompt_tokens = statistics.mean(pts) if pts else float("nan")
            avg_completion_tokens = statistics.mean(cts) if cts else float("nan")
            avg_total_tokens = statistics.mean(tts) if tts else float("nan")
            # 초당 토큰수 및 p95 토큰수 계산 (output throughput)
            avg_tokens_per_second = statistics.mean(tps) if tps else float("nan")
            p95_tokens_per_second = sorted(tps)[max(0, int(len(tps) * 0.95) - 1)] if tps else float("nan")

            rows.append(
                {
                    "mode": getattr(run_benchmark, "_prompt_mode", "strict"),
                    "model": model,
                    "repeat": repeat,
                    "ok_rate": f"{ok_cnt}/{repeat}",
                    "rag_on_rate": f"{rag_on_cnt}/{repeat}",
                    "avg_retrieved_coaching_n": f"{avg_retrieved_coaching_n:.2f}" if not math.isnan(float(avg_retrieved_coaching_n)) else "",
                    "avg_score": f"{avg_score:.2f}" if not math.isnan(float(avg_score)) else "",
                    "avg_latency_ms": f"{avg_latency:.1f}" if not math.isnan(float(avg_latency)) else "",
                    "p95_latency_ms": f"{p95_latency:.1f}" if not math.isnan(float(p95_latency)) else "",
                    "avg_tokens_per_second": f"{avg_tokens_per_second:.2f}" if not math.isnan(float(avg_tokens_per_second)) else "",
                    "p95_tokens_per_second": f"{p95_tokens_per_second:.2f}" if not math.isnan(float(p95_tokens_per_second)) else "",
                    "avg_prompt_tokens": f"{avg_prompt_tokens:.1f}" if not math.isnan(float(avg_prompt_tokens)) else "",
                    "avg_completion_tokens": f"{avg_completion_tokens:.1f}" if not math.isnan(float(avg_completion_tokens)) else "",
                    "avg_total_tokens": f"{avg_total_tokens:.1f}" if not math.isnan(float(avg_total_tokens)) else "",
                }
            )

        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "mode",
                    "model",
                    "repeat",
                    "ok_rate",
                    "rag_on_rate",
                    "avg_retrieved_coaching_n",
                    "avg_score",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "avg_tokens_per_second",
                    "p95_tokens_per_second",
                    "avg_prompt_tokens",
                    "avg_completion_tokens",
                    "avg_total_tokens",
                ]
            )
            w.writeheader()
            w.writerows(rows)

        print(f"[OK] CSV saved: {csv_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["strict", "natural"],
        default="strict",
        help="프롬프트 모드 선택 (strict=배포 기본, natural=완화된 비교 리포트)",
    )

    # A: fetch + save
    ap.add_argument("--fetch-meta", action="store_true", help="API로 meta를 생성/저장")
    ap.add_argument("--api-base", default="http://localhost:8000")
    ap.add_argument("--post-idx", default="user_001")
    ap.add_argument("--range", dest="range_", default="7d")
    ap.add_argument("--out", default="snapshots/meta_sample.json")

    # B: run benchmark from fixed meta file
    ap.add_argument("--meta", default="snapshots/meta_sample.json", help="고정 meta JSON 파일")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--csv-out", default="", help="모델별 avg_score/latency CSV 저장 경로 (예: snapshots/bench.csv)")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--freeze-rag", action="store_true", help="retrieved_coaching을 1회 계산해 meta에 고정")

    args = ap.parse_args()

    # --- Load .env automatically for local runs (optional dependency) ---
    # IMPORTANT: CHROMA_DIR/COACH_KB_PATH는 LLM_Total_report import 시점에 읽히므로,
    # 벤치마크 스크립트에서는 가능한 한 빨리 .env를 로딩해야 합니다.
    try:
        from dotenv import load_dotenv  # type: ignore

        # NOTE:
        # - 보통 `backend/` 폴더에서 실행하므로, 여기서의 `.env`는 곧 `backend/.env` 입니다.
        # - `.env`와 `backend/.env`를 동시에 로딩하면 값이 섞여 디버깅이 어려워집니다.
        #   따라서 현재 작업 디렉토리의 `.env`만 로딩합니다.
        load_dotenv(dotenv_path=".env", override=False)
    except Exception:
        # If python-dotenv isn't installed (or import fails), we'll handle below.
        pass

    if args.freeze_rag:
        print(
            f"[INFO] CHROMA_DIR={os.getenv('CHROMA_DIR')} "
            f"CHROMA_COLLECTION={os.getenv('CHROMA_COLLECTION') or 'coach_kb'}"
        )

    # NOTE: --fetch-meta only calls your local API and does not require GROQ.
    # NOTE: --fetch-meta only calls your local API and does not require any LLM key.
    if not args.fetch_meta:
        provider = (os.getenv("LLM_PROVIDER", "groq") or "groq").strip().lower()
        if provider == "hf":
            if not os.getenv("HF_BASE_URL"):
                raise RuntimeError(
                    "HF_BASE_URL 환경변수가 필요합니다. (예: https://<your-endpoint>/v1)\n"
                    "현재 작업 디렉토리의 .env를 로딩하며, 값이 비어있으면 발생할 수 있습니다."
                )
            if not (os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")):
                raise RuntimeError(
                    "HF_API_KEY (또는 HF_TOKEN) 환경변수가 필요합니다.\n"
                    "현재 작업 디렉토리의 .env를 로딩하며, 값이 비어있으면 발생할 수 있습니다."
                )
        else:
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError(
                    "GROQ_API_KEY 환경변수가 필요합니다.\n"
                    "- (권장) python -m pip install python-dotenv 후 다시 실행하거나\n"
                    "- 터미널에서 export GROQ_API_KEY=... 로 환경변수를 설정하세요.\n"
                    "(이 스크립트는 현재 작업 디렉토리의 .env를 로딩합니다. .env가 없거나 값이 비어있으면 발생할 수 있습니다.)"
                )

    if args.fetch_meta:
        meta = fetch_meta_from_api(args.api_base, args.post_idx, args.range_, args.lang)
        save_json(args.out, meta)
        if isinstance(meta, dict) and meta.get("__warning"):
            print("[WARN] API 응답에 meta가 포함되지 않아 응답 전체를 저장했습니다.")
            print("reportRouters.py의 /api/report/post/{post_idx} 에 snapshot_only/debug_meta 옵션이 적용됐는지 확인하세요.")
        print(f"[OK] meta saved: {args.out}")
        return

    # run benchmark
    meta = json.load(open(args.meta, "r", encoding="utf-8"))

    # 만약 fetch-meta로 저장했는데 meta가 아니라 response wrapper면, meta 본문을 찾기 어렵습니다.
    # 이 경우 사용자에게 안내하고 종료합니다.
    if isinstance(meta, dict) and "response" in meta and meta.get("__warning"):
        print("[ERROR] meta 파일에 실제 meta가 아니라 API 응답 전체가 저장되어 있습니다.")
        print("        reportRouters.py가 meta를 반환하도록 수정한 뒤 --fetch-meta를 다시 실행하세요.")
        return

    if args.freeze_rag:
        meta = ensure_frozen_rag(meta)
        # freeze된 meta를 덮어써서 재현성 확보
        save_json(args.meta, meta)
        print(f"[OK] frozen meta(updated): {args.meta} (rag={'on' if (meta.get('retrieved_coaching') or []) else 'off'})")

    # 전달된 --csv-out 이 있으면 벤치마크 함수에 주입(전역 상태 최소화)
    setattr(run_benchmark, "_csv_out_override", args.csv_out.strip() or "")
    # 모드 주입
    setattr(run_benchmark, "_prompt_mode", args.mode)
    run_benchmark(meta, args.models, args.repeat, args.lang)


if __name__ == "__main__":
    main()