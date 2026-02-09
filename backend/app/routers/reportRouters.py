from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.report.LLM_report import generate_report

router = APIRouter(prefix="/api/report", tags=["Report"])


class PostureReportRequest(BaseModel):
    angles: Dict[str, float] = Field(..., description="관절별 오차각도")
    meta: Optional[Dict[str, Any]] = Field(default=None)
    lang: str = Field(default="ko")


class PostureReportResponse(BaseModel):
    report: Dict[str, Any]


@router.post("/posture", response_model=PostureReportResponse)
def posture_report(payload: PostureReportRequest):
    try:
        report = generate_report(
            angles=payload.angles,
            meta=payload.meta,
            lang=payload.lang,
        )
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))