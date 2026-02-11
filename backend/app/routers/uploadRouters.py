"""
영상 업로드 분석 API
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.swing.video_analysis_service import video_analysis_service

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/video")
async def upload_video(
    user_id: str,
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    영상 업로드 및 분석
    
    Args:
        user_id: 사용자 ID
        video: 영상 파일 (.mp4, .avi, .mov, 5초 이내)
    
    Returns:
        post_idx: 생성된 POST ID
        status: DONE
        total_score: 종합 점수
        message: 완료 메시지
    """
    
    try:
        # 파일 검증
        if not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400, 
                detail="영상 파일만 업로드 가능합니다."
            )
        
        # 파일 크기 제한 (50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        content = await video.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail="파일 크기는 50MB 이하여야 합니다."
            )
        
        # 파일 포인터 리셋
        await video.seek(0)
        
        # 분석 시작
        result = await video_analysis_service.analyze_video(user_id, video, db)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@router.get("/status/{post_idx}")
async def get_analysis_status(
    post_idx: str,
    db: Session = Depends(get_db)
):
    """
    분석 상태 확인
    
    Returns:
        post_idx: POST ID
        status: ANALYZING / DONE
        progress: 진행률 (0~100)
    """
    
    try:
        status = await video_analysis_service.get_status(post_idx, db)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{post_idx}")
async def get_analysis_result(
    post_idx: str,
    db: Session = Depends(get_db)
):
    """
    분석 결과 조회
    
    Returns:
        post_idx: POST ID
        total_score: 종합 점수
        scores: 6대 지표 점수
        keyframes: 키프레임 파일 정보
        feedback: 피드백 메시지
    """
    
    try:
        result = await video_analysis_service.get_result(post_idx, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))