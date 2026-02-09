from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import os
import pandas as pd
from datetime import datetime

from app.services.mediapipe_service import MediaPipeService
from app.crud import userCrud, postCrud, analysisCrud, fileCrud
from app.db.data import get_db
from app.db.base import Base
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter(prefix="/api/realtime", tags=["Realtime Analysis"])

# MediaPipe 서비스 초기화
mp_service = MediaPipeService()

# 현재 진행 중인 스윙 데이터 임시 저장 (메모리)
swing_sessions = {}


class FrameData(BaseModel):
    """프레임 데이터 구조"""
    frame_id: int
    image: str  # Base64


class SwingRequest(BaseModel):
    """스윙 분석 요청"""
    user_id: str
    swing_num: int  # 1, 2, 3
    frames: List[FrameData]  # 90개 프레임


@router.post("/analyze-swing")
async def analyze_swing(
    request: SwingRequest,
    db: Session = Depends(get_db)
):
    """
    스윙 1회 분석
    
    Args:
        request: {
            user_id: "user_001",
            swing_num: 1,
            frames: [
                {frame_id: 0, image: "data:image/jpeg;base64,..."},
                ...
            ]
        }
    
    Returns:
        {
            post_idx: "uuid",
            swing_num: 1,
            keyframes: {
                ready: 30,
                backswing: 48,
                impact: 60
            },
            scores: {
                ready: 85.5,
                backswing: 72.3,
                impact: 90.1
            },
            overall_average: 82.6,
            feedback: "팔꿈치를 더 높이세요!"
        }
    """
    
    try:
        # 1. POST 생성
        post_idx = str(uuid.uuid4())
        post_data = {
            "idx": post_idx,
            "user_id": request.user_id,
            "type": "REALTIME",
            "swing_num": request.swing_num,
            "status": "ANALYZING"
        }
        postCrud.create_post(db, post_data)
        
        # 2. MediaPipe keypoint 추출
        keypoints_list = []
        
        for frame_data in request.frames:
            keypoints = mp_service.extract_keypoints_from_base64(frame_data.image)
            
            if keypoints:
                keypoints['frame_id'] = frame_data.frame_id
                keypoints_list.append(keypoints)
        
        if not keypoints_list:
            raise HTTPException(status_code=400, detail="No keypoints detected")
        
        # 3. CSV 저장 (임시)
        df = pd.DataFrame(keypoints_list)
        csv_path = f"backend/data/video/{post_idx}_keypoints.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        # 4. 키프레임 감지 (기존 알고리즘 import)
        from backend.swing.keyframe_detector import analyze_swing_keyframes
        
        kf_result_csv = f"backend/data/video/{post_idx}_keyframes.csv"
        kf_result = analyze_swing_keyframes(csv_path, kf_result_csv)
        
        if not kf_result:
            raise HTTPException(status_code=500, detail="Keyframe detection failed")
        
        # 5. 점수 계산 (기존 알고리즘 import)
        from backend.swing.score_calculator import BadmintonSuperStrictCalculator
        
        calculator = BadmintonSuperStrictCalculator(
            "backend/data/standard/GT_angle/gt_total_metrics2.csv"
        )
        
        score_json_path = f"backend/data/video/{post_idx}_scores.json"
        calculator.calculate_final_scores(
            csv_path,
            kf_result_csv,
            score_json_path
        )
        
        # 6. JSON 로드
        import json
        with open(score_json_path, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        # 7. ANALYSIS 저장
        analysis_data = {
            "idx": str(uuid.uuid4()),
            "post_idx": post_idx,
            "kf1": kf_result['ready'],
            "kf2": kf_result['backswing'],
            "kf3": kf_result['impact'],
            "score_json": score_data
        }
        analysisCrud.create_analysis(db, analysis_data)
        
        # 8. 키프레임 이미지 저장 (FILE 테이블)
        # TODO: 이미지 저장 로직 (다음 단계)
        
        # 9. POST 업데이트
        postCrud.update_post_status(db, post_idx, "DONE", score_data['overall_average'])
        
        # 10. 피드백 생성
        feedback = generate_quick_feedback(score_data)
        
        # 11. 응답
        return JSONResponse({
            "post_idx": post_idx,
            "swing_num": request.swing_num,
            "keyframes": kf_result,
            "scores": {
                item['단계']: item['점수'] 
                for item in score_data['user_evaluation']
            },
            "overall_average": score_data['overall_average'],
            "feedback": feedback
        })
        
    except Exception as e:
        print(f"Error in analyze_swing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_quick_feedback(score_data: dict) -> str:
    """
    점수 기반 빠른 피드백 생성
    
    Args:
        score_data: {
            "user_evaluation": [
                {"단계": "ready", "점수": 85.5, ...},
                ...
            ],
            "overall_average": 82.6
        }
    
    Returns:
        "팔꿈치를 더 높이세요!"
    """
    
    avg = score_data['overall_average']
    evaluations = score_data['user_evaluation']
    
    # 가장 낮은 점수 찾기
    lowest = min(evaluations, key=lambda x: x['점수'])
    
    feedbacks = {
        'ready': [
            "준비 자세가 아쉬워요. 팔꿈치를 어깨보다 높이!",
            "준비 자세 연습이 필요해요.",
        ],
        'backswing': [
            "백스윙을 더 크게 해보세요!",
            "팔꿈치를 더 높이 올려보세요!",
            "백스윙 각도를 늘려보세요!",
        ],
        'impact': [
            "임팩트 순간 팔을 완전히 펴세요!",
            "타구 위치를 앞으로!",
            "몸통 회전이 부족해요!",
        ]
    }
    
    if avg >= 90:
        return "완벽합니다! 👍"
    elif avg >= 70:
        stage = lowest['단계']
        return feedbacks.get(stage, ["조금만 더!"])[0]
    else:
        return "자세 교정이 많이 필요해요. 천천히 연습해봐요!"