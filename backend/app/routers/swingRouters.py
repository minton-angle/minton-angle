"""
Swing Analysis API Router (얇게!)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Union

from app.db.session import get_db
from app.models.userModels import User
from app.schemas.swing import (
    SwingAnalysisRequest,
    QuickFeedbackResponse,
    AnalysisCompleteResponse
)
from app.services.swing.swing_service import swing_service

from app.routers.authRouters import get_current_user

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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # ⭐ User 객체로 받음!
):
    """실시간 스윙 분석 - Service에 위임"""
    
    try:
        # ⭐ user_id를 current_user에서 추출하여 서비스로 전달
        result = await swing_service.analyze_realtime(
            request=request,
            db=db,
            user_id=current_user.id  # ⭐ 여기서 전달!
        )
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"❌ 분석 에러: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 실패: {str(e)}"
        )