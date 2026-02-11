"""
Swing Analysis API Router (얇게!)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Union

from app.db.session import get_db
from app.schemas.swing import (
    SwingAnalysisRequest,
    QuickFeedbackResponse,
    AnalysisCompleteResponse
)
from app.services.swing.swing_service import swing_service

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


@router.post(
    "/analyze-swing",
    response_model=Union[QuickFeedbackResponse, AnalysisCompleteResponse],
    summary="실시간 스윙 분석",
    description="""
    실시간 스윙 분석 API
    
    **플로우:**
    1. 1회차: POST 생성 → post_id 반환
    2. 2회차: POST 업데이트 (post_id 필수)
    3. 3회차: POST 업데이트 + 완료
    
    **모든 회차에서 DB 저장됨**
    """
)
async def analyze_swing(
    request: SwingAnalysisRequest,
    db: Session = Depends(get_db)
):
    """실시간 스윙 분석 - Service에 위임"""
    
    try:
        result = await swing_service.analyze_realtime(request, db)
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 실패: {str(e)}"
        )
