from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from sqlalchemy.orm import Session
import uuid
import os
import pandas as pd

from app.db.session import get_db  # 🆕 수정!
from app.crud import postCrud, analysisCrud, fileCrud
from app.services.swing.engine import PoseDetector, KeyframeDetector, ScoreCalculator

router = APIRouter(prefix="/api/realtime", tags=["Realtime Analysis"])

# 서비스 초기화
pose_detector = PoseDetector()
keyframe_detector = KeyframeDetector()

# GT 기준값 경로 (실제 경로로 수정 필요)
GT_METRICS_PATH = "backend/data/standard/GT_angle/gt_total_metrics2.csv"


class FrameData(BaseModel):
    """프레임 데이터 구조"""
    frame_id: int
    image: str  # Base64 이미지


class SwingRequest(BaseModel):
    """스윙 분석 요청"""
    user_id: str
    swing_num: int  # 1, 2, 3
    frames: List[FrameData]


@router.post("/analyze-swing")
async def analyze_swing(
    request: SwingRequest,
    db: Session = Depends(get_db)
):
    """
    스윙 1회 분석
    
    프로세스:
    1. POST 생성 (DB)
    2. MediaPipe로 keypoint 추출
    3. CSV 저장
    4. 키프레임 감지
    5. 점수 계산
    6. ANALYSIS 저장 (DB)
    7. 피드백 생성
    8. 응답
    """
    
    try:
        print(f"\n{'='*50}")
        print(f"🎯 스윙 {request.swing_num}회 분석 시작")
        print(f"{'='*50}")
        
        # --- [1] POST 생성 ---
        post_idx = str(uuid.uuid4())
        post_data = {
            "idx": post_idx,
            "user_id": request.user_id,
            "type": "REALTIME",
            "swing_num": request.swing_num,
            "status": "ANALYZING"
        }
        postCrud.create_post(db, post_data)
        print(f"✅ POST 생성: {post_idx}")
        
        # # --- [2] MediaPipe keypoint 추출 ---
        # print(f"🔍 프레임 {len(request.frames)}개에서 keypoint 추출 중...")
        
        # keypoints_list = []
        # for frame_data in request.frames:
        #     keypoints = pose_detector.extract_from_base64(frame_data.image)
            
        #     if keypoints:
        #         keypoints['frame_id'] = frame_data.frame_id
        #         keypoints_list.append(keypoints)
        
        # if not keypoints_list:
        #     raise HTTPException(status_code=400, detail="No keypoints detected")
        
        # print(f"✅ Keypoint 추출 완료: {len(keypoints_list)}개 프레임")
        
        # --- [2] MediaPipe keypoint 추출 ---
        print(f"🔍 프레임 {len(request.frames)}개에서 keypoint 추출 중...")

        keypoints_list = []

        # 🆕 첫 프레임 디버깅
        if request.frames:
            first_frame = request.frames[0]
            print(f"📸 첫 프레임 길이: {len(first_frame.image)} bytes")
            print(f"📸 첫 프레임 시작: {first_frame.image[:50]}")  # 처음 50자

        for frame_data in request.frames:
            keypoints = pose_detector.extract_from_base64(frame_data.image)
            
            if keypoints:
                keypoints['frame_id'] = frame_data.frame_id
                keypoints_list.append(keypoints)
            else:
                # 🆕 실패 로그 추가
                print(f"⚠️ 프레임 {frame_data.frame_id}: keypoint 추출 실패")

        print(f"✅ Keypoint 추출 완료: {len(keypoints_list)}개 프레임")
        # --- [3] CSV 저장 ---
        df = pd.DataFrame(keypoints_list)
        
        # 디렉토리 생성
        csv_dir = "backend/data/video"
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_path = f"{csv_dir}/{post_idx}_keypoints.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV 저장: {csv_path}")
        
        # --- [4] 키프레임 감지 ---
        print(f"🔍 키프레임 감지 중...")
        keyframes = keyframe_detector.detect(df)
        
        if not keyframes:
            raise HTTPException(status_code=500, detail="Keyframe detection failed")
        
        print(f"✅ 키프레임: {keyframes}")
        
        # --- [5] 점수 계산 ---
        print(f"📊 점수 계산 중...")
        
        if not os.path.exists(GT_METRICS_PATH):
            raise HTTPException(
                status_code=500, 
                detail=f"GT 기준값 파일 없음: {GT_METRICS_PATH}"
            )
        
        calculator = ScoreCalculator(GT_METRICS_PATH)
        score_data = calculator.calculate_scores(df, keyframes)
        
        print(f"✅ 점수 계산 완료: {score_data['overall_average']}점")
        
        # --- [6] ANALYSIS 저장 ---
        analysis_data = {
            "idx": str(uuid.uuid4()),
            "post_idx": post_idx,
            "kf1": keyframes['ready'],
            "kf2": keyframes['backswing'],
            "kf3": keyframes['impact'],
            "kf1_error": 0.0,  # TODO: 나중에 계산
            "kf2_error": 0.0,
            "kf3_error": 0.0,
            "score_json": score_data
        }
        analysisCrud.create_analysis(db, analysis_data)
        print(f"✅ ANALYSIS 저장 완료")
        
        # --- [7] POST 상태 업데이트 ---
        postCrud.update_post_status(
            db, 
            post_idx, 
            "DONE", 
            int(score_data['overall_average'])
        )
        print(f"✅ POST 상태: DONE")
        
        # --- [8] 빠른 피드백 생성 ---
        feedback = generate_quick_feedback(score_data)
        
        # --- [9] 응답 ---
        response = {
            "post_idx": post_idx,
            "swing_num": request.swing_num,
            "keyframes": keyframes,
            "scores": {
                item['단계']: item['점수'] 
                for item in score_data['user_evaluation']
            },
            "overall_average": score_data['overall_average'],
            "feedback": feedback,
            "save_to_db": True
        }
        
        print(f"\n{'='*50}")
        print(f"✅ 스윙 {request.swing_num}회 분석 완료!")
        print(f"총점: {score_data['overall_average']}점")
        print(f"피드백: {feedback}")
        print(f"{'='*50}\n")
        
        return JSONResponse(response)
        
    except Exception as e:
        print(f"❌ Error in analyze_swing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def generate_quick_feedback(score_data: Dict) -> str:
    """
    점수 기반 빠른 피드백 생성
    """
    avg = score_data['overall_average']
    evaluations = score_data['user_evaluation']
    
    # 가장 낮은 점수 찾기
    lowest = min(evaluations, key=lambda x: x['점수'])
    
    feedbacks = {
        'ready': [
            "준비 자세가 아쉬워요. 팔꿈치를 어깨보다 높이 올려보세요!",
            "준비 동작 연습이 필요해요.",
        ],
        'backswing': [
            "백스윙을 더 크게 해보세요!",
            "팔꿈치를 더 높이 올려보세요!",
            "백스윙 각도를 늘려보세요!",
        ],
        'impact': [
            "임팩트 순간 팔을 완전히 펴세요!",
            "몸통 회전이 부족해요!",
            "골반을 더 회전시켜보세요!",
        ]
    }
    
    if avg >= 90:
        return "완벽합니다! 프로 수준이에요! 👍"
    elif avg >= 80:
        return "아주 잘하고 있어요! 조금만 더 연습하면 완벽해요! 💪"
    elif avg >= 70:
        stage = lowest['단계']
        return feedbacks.get(stage, ["조금만 더 노력해요!"])[0]
    elif avg >= 50:
        stage = lowest['단계']
        return feedbacks.get(stage, ["많은 연습이 필요해요!"])[1]
    else:
        return "기초부터 천천히 다시 배워봐요! 포기하지 마세요! 🔥"


@router.get("/health")
def health_check():
    """헬스체크"""
    return {
        "status": "healthy",
        "service": "realtime analysis",
        "gt_metrics": os.path.exists(GT_METRICS_PATH)
    }