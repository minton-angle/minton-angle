from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport


VALID_RANGES = {"7d", "1m", "3m", "all", "5n", "10n"}


def normalize_report_range(value: str | None) -> str:
    """Normalize report range aliases into the canonical range values."""
    r = (value or "7d").lower().strip()
    if r in ("5n", "last5", "recent5"):
        return "5n"
    if r in ("10n", "last10", "recent10"):
        return "10n"
    return r if r in VALID_RANGES else "7d"


def _current_start_for_range(r: str, now: datetime) -> Optional[datetime]:
    if r == "7d":
        return now - timedelta(days=7)
    if r == "1m":
        return now - timedelta(days=30)
    if r == "3m":
        return now - timedelta(days=90)
    return None


def load_analysis_windows(
    db: Session,
    post_idx: str,
    range_value: str | None = "7d",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Load current and previous Analysis windows for a post.

    Responsibilities:
    - range normalization
    - recent-N window loading for 5n/10n
    - date-based current/previous window loading for 7d/1m/3m/all
    - ascending order normalization for chart/report usage
    """
    r = normalize_report_range(range_value)
    now = now or datetime.utcnow()
    base_q = db.query(Analysis).filter(Analysis.post_idx == post_idx)

    if r in ("5n", "10n"):
        n = 5 if r == "5n" else 10
        latest = base_q.order_by(Analysis.create_date.desc()).limit(2 * n).all()

        cur_desc = latest[:n]
        prev_desc = latest[n:2 * n]

        current_analyses = list(reversed(cur_desc))
        prev_analyses = list(reversed(prev_desc))

        return {
            "range": r,
            "current_analyses": current_analyses,
            "prev_analyses": prev_analyses,
            "current_start": None,
            "previous_start": None,
            "previous_end": None,
        }

    current_start = _current_start_for_range(r, now)

    q_current = base_q
    if current_start is not None:
        q_current = q_current.filter(Analysis.create_date >= current_start)

    current_analyses = q_current.order_by(Analysis.create_date.asc()).all()

    prev_analyses: list[Analysis] = []
    prev_start = None
    prev_end = None
    if current_start is not None:
        window_days = (now - current_start).days
        prev_start = current_start - timedelta(days=window_days)
        prev_end = current_start

        q_prev = base_q.filter(
            Analysis.create_date >= prev_start,
            Analysis.create_date < prev_end,
        )
        prev_analyses = q_prev.order_by(Analysis.create_date.asc()).all()

    return {
        "range": r,
        "current_analyses": current_analyses,
        "prev_analyses": prev_analyses,
        "current_start": current_start,
        "previous_start": prev_start,
        "previous_end": prev_end,
    }


def load_latest_llm_report(db: Session, post_idx: str) -> Optional[LLMReport]:
    """Load the most recent saved LLM report for a post."""
    return (
        db.query(LLMReport)
        .filter(LLMReport.post_idx == post_idx)
        .order_by(LLMReport.create_date.desc())
        .first()
    )


def latest_llm_report_payload(llm_report: Optional[LLMReport]) -> Optional[Dict[str, Any]]:
    """Serialize latest LLMReport for API response."""
    if llm_report is None:
        return None
    return {
        "idx": llm_report.idx,
        "created_at": llm_report.create_date.isoformat() if llm_report.create_date else None,
        "report": llm_report.feedback,
    }