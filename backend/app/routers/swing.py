"""
Swing Analysis API Router
스윙 분석 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Union
from datetime import datetime
import uuid
import os
import base64

# 현재는 상대 경로로 임포트 (나중에 절대 경로로 변경 가능)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import get_db  # DB 세션
from app.models.postModels import Post  # POST 모델
from app.models.fileModels import File  # FILE 모델
from app.models.analysisModels import Analysis  # ANALYSIS 모델
from app.models.llmReportModels import LLMReport  # LLM_REPORT 모델

# 스키마
from schemas.swing import (
    SwingAnalysisRequest,
    QuickFeedbackResponse,
    AnalysisCompleteResponse,
    ScoreDetail
)

# 서비스
from services.swing_service import swing_service


router = APIRouter(prefix="/api/swing", tags=["swing"])


# ==================== 헬퍼 함수 ====================

def save_keyframe_image(
    post_id: str,
    kf_num: int,
    swing_num: int,
    image_base64: str,
    save_dir: str = "/tmp/keyframes"
) -> str:
    """
    키프레임 이미지 저장
    
    Args:
        post_id: POST ID
        kf_num: 키프레임 번호 (1, 2, 3)
        swing_num: 스윙 횟수 (1, 2, 3)
        image_base64: Base64 인코딩된 이미지
        save_dir: 저장 디렉토리
        
    Returns:
        파일 경로
    """
    # 디렉토리 생성
    post_dir = os.path.join(save_dir, post_id)
    os.makedirs(post_dir, exist_ok=True)
    
    # Base64 디코딩
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    try:
        image_data = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")
    
    # 이미지 저장
    filename = f"swing{swing_num}_kf{kf_num}.jpg"
    filepath = os.path.join(post_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filepath


# ==================== API 엔드포인트 ====================

@router.post(
    "/analyze",
    response_model=Union[QuickFeedbackResponse, AnalysisCompleteResponse],
    summary="실시간 스윙 분석",
    description="""
    실시간 스윙 분석 API
    
    **플로우:**
    1. 1회차: POST 생성 → post_id 반환
    2. 2회차: POST 업데이트 (post_id 필수)
    3. 3회차: POST 업데이트 + LLM_REPORT 생성
    
    **모든 회차에서 DB 저장됨**
    """
)
async def analyze_swing(
    request: SwingAnalysisRequest,
    db: Session = Depends(get_db)
):
    """실시간 스윙 분석"""
    
    # 1. 키프레임 감지
    kf1, kf2, kf3 = swing_service.detect_keyframes(request.keypoints)
    
    # 2. 점수 계산
    scores = swing_service.calculate_scores(request.keypoints, kf1, kf2, kf3)
    
    # 3. 빠른 피드백 생성
    quick_feedback = swing_service.get_quick_feedback(scores)
    
    # 4. 종합 점수 계산
    total_score = sum(scores.values()) // len(scores)
    
    try:
        # ==================== 1회차: 새 POST 생성 ====================
        if request.swing_num == 1:
            if request.post_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="1회차에서는 post_id를 보내면 안 됩니다."
                )
            
            # POST 생성
            post_id = str(uuid.uuid4())
            post = Post(
                idx=post_id,
                user_id=request.user_id,
                type="REALTIME",
                status="ANALYZING",
                total_score=total_score
            )
            db.add(post)
            db.flush()
            
            # ANALYSIS 생성
            analysis_id = str(uuid.uuid4())
            analysis = Analysis(
                idx=analysis_id,
                post_idx=post_id,
                kf1=kf1,
                kf2=kf2,
                kf3=kf3,
                score_json=scores
            )
            db.add(analysis)
            
            # FILE 생성 (키프레임 이미지)
            if request.frames and len(request.frames) > max(kf1, kf2, kf3):
                for kf_num, kf_idx in enumerate([kf1, kf2, kf3], 1):
                    frame_image = request.frames[kf_idx]
                    file_path = save_keyframe_image(post_id, kf_num, 1, frame_image)
                    
                    file = File(
                        idx=str(uuid.uuid4()),
                        post_idx=post_id,
                        file_type=f"KF{kf_num}",
                        file_name=f"swing1_kf{kf_num}.jpg",
                        file_path=file_path,
                        file_extension="jpg",
                        storage_type="LOCAL"
                    )
                    db.add(file)
            
            db.commit()
            db.refresh(post)
            
            return QuickFeedbackResponse(
                swing_num=1,
                post_id=post_id,
                quick_feedback=quick_feedback,
                save_to_db=True,
                scores=ScoreDetail(**scores)
            )
        
        # ==================== 2~3회차: 기존 POST 업데이트 ====================
        else:
            if not request.post_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="2~3회차에서는 post_id가 필수입니다."
                )
            
            # POST 조회
            post = db.query(Post).filter(
                Post.idx == request.post_id,
                Post.user_id == request.user_id
            ).first()
            
            if not post:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="POST를 찾을 수 없습니다. post_id를 확인하세요."
                )
            
            # POST 업데이트 (평균 점수로)
            post.total_score = total_score
            
            # ANALYSIS 조회 및 업데이트
            analysis = db.query(Analysis).filter(
                Analysis.post_idx == request.post_id
            ).first()
            
            if analysis:
                # 점수 업데이트 (평균 계산)
                old_scores = analysis.score_json
                new_scores = {}
                for key in scores:
                    # 기존 점수와 새 점수의 평균
                    new_scores[key] = (old_scores.get(key, 0) + scores[key]) // 2
                
                analysis.score_json = new_scores
                analysis.kf1 = kf1
                analysis.kf2 = kf2
                analysis.kf3 = kf3
            
            # FILE 추가 (새 키프레임 이미지)
            if request.frames and len(request.frames) > max(kf1, kf2, kf3):
                for kf_num, kf_idx in enumerate([kf1, kf2, kf3], 1):
                    frame_image = request.frames[kf_idx]
                    file_path = save_keyframe_image(
                        request.post_id, 
                        kf_num, 
                        request.swing_num, 
                        frame_image
                    )
                    
                    file = File(
                        idx=str(uuid.uuid4()),
                        post_idx=request.post_id,
                        file_type=f"KF{kf_num}",
                        file_name=f"swing{request.swing_num}_kf{kf_num}.jpg",
                        file_path=file_path,
                        file_extension="jpg",
                        storage_type="LOCAL"
                    )
                    db.add(file)
            
            # ==================== 3회차: LLM_REPORT 생성 ====================
            if request.swing_num == 3:
                # 상세 피드백 생성
                detailed_feedback = swing_service.generate_detailed_feedback(
                    scores=analysis.score_json if analysis else scores,
                    kf1=kf1,
                    kf2=kf2,
                    kf3=kf3
                )
                
                # LLM_REPORT 생성
                llm_report = LLMReport(
                    idx=str(uuid.uuid4()),
                    post_idx=request.post_id,
                    summary=detailed_feedback.get("overall", "분석 완료"),
                    feedback=detailed_feedback
                )
                db.add(llm_report)
                
                # POST 상태 완료로 변경
                post.status = "DONE"
                post.total_score = sum(analysis.score_json.values()) // len(analysis.score_json)
                
                db.commit()
                db.refresh(post)
                
                return AnalysisCompleteResponse(
                    swing_num=3,
                    post_id=request.post_id,
                    save_to_db=True,
                    total_score=post.total_score,
                    scores=ScoreDetail(**analysis.score_json),
                    quick_feedback=quick_feedback
                )
            
            # 2회차는 빠른 피드백만
            else:
                db.commit()
                db.refresh(post)
                
                return QuickFeedbackResponse(
                    swing_num=2,
                    post_id=request.post_id,
                    quick_feedback=quick_feedback,
                    save_to_db=True,
                    scores=ScoreDetail(**scores)
                )
    
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 처리 중 오류 발생: {str(e)}"
        )