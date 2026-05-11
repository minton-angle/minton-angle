from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.models.llmReportModels import LLMReport
from app.services.report.LLM_Total_report import generate_report


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
    report = generate_report(angles={}, meta=meta, lang=lang)

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