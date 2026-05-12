from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.models.llmReportModels import LLMReport
from app.services.report.LLM_Total_report import generate_report

from app.services.report.agent.nodes import movement_reasoning_node
from app.services.report.agent.state import ReportAgentState


def build_initial_report_agent_state(meta: Dict[str, Any]) -> ReportAgentState:
    """Build the initial state for the report reasoning workflow.

    The router/service layer already prepares score_stats and weak_metrics.
    This helper converts that meta payload into the explicit state shape used
    by agent nodes.
    """
    meta = meta or {}
    return {
        "meta": meta,
        "score_stats": meta.get("score_stats", {}) or {},
        "weak_metrics": meta.get("weak_metrics", []) or [],
    }


def upgrade_meta_with_movement_reasoning(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Run movement_reasoning_node and attach its result to meta.

    This is the first real connection point between the refactored report
    pipeline and the agentic reasoning workflow.
    """
    state = build_initial_report_agent_state(meta)
    next_state = movement_reasoning_node(state)

    upgrade_meta = dict(meta or {})
    upgrade_meta["movement_reasoning"] = next_state.get("movement_reasoning", {})
    return upgrade_meta


def create_and_save_llm_report(
    *,
    db: Session,
    post_idx: str,
    meta: Dict[str, Any],
    lang: str = "ko",
) -> Dict[str, Any]:
    """Generate an LLM coaching report and persist it to llm_report.

    Router responsibility should remain API orchestration only.
    This service owns:
    - calling the LLM/RAG report generation pipeline
    - creating the LLMReport ORM row
    - committing the generated report JSON
    """
    upgrade_meta = upgrade_meta_with_movement_reasoning(meta)
    report = generate_report(angles={}, meta=upgrade_meta, lang=lang)

    llm_row = LLMReport(
        idx=str(uuid.uuid4()),
        post_idx=post_idx,
        feedback=report,
        create_date=datetime.utcnow(),
    )
    db.add(llm_row)
    db.commit()

    return {
        "report": report,
        "llm_report_idx": llm_row.idx,
    }