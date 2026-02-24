"""
영상 업로드 라우터
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.postModels import Post
from app.models.fileModels import File as FileModel
from app.models.analysisModels import Analysis
from app.services.swing.video_analysis_service import VideoAnalysisService, video_analysis_service

from app.routers.userRouters import get_current_user
from app.models.userModels import User

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/video")
async def upload_video(
    #current_user = Depends(get_current_user),
    user_id: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    
):
    """영상 업로드 및 분석"""
    
    try:
        # ⭐ video_analysis_service.video_analysis_service 사용
        # user_id = current_user.id
        result = await video_analysis_service.analyze_video(user_id, video, db)
        print(f"📡 프론트엔드로 보낼 최종 데이터: {result}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{post_idx}")
async def get_upload_status(post_idx: str, db: Session = Depends(get_db)):
    """분석 상태 조회"""
    
    try:
        result = await video_analysis_service.get_status(post_idx, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{post_idx}")
async def get_analysis_result(post_idx: str, db: Session = Depends(get_db)):
    """리포트용 분석 결과 조회"""
    
    try:
        # POST 조회
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
        
        # ANALYSIS 조회
        analysis = db.query(Analysis).filter(Analysis.post_idx == post_idx).first()
        
        # FILE 조회
        files = db.query(FileModel).filter(FileModel.post_idx == post_idx).all()
        
        print(f"\n{'='*50}")
        print(f"📊 결과 조회: {post_idx}")
        print(f"파일 개수: {len(files)}")
        
       # 파일 타입별 경로 매핑
        file_paths = {}
        for file in files:
            # DB에 저장된 원래 경로 (예: /app/data/upload_keyframes/...)
            raw_path = file.file_path
            
            print(f"   [원본 경로] {file.file_type}: {raw_path}")

            # 도커 내부 절대경로(/app/data)를 브라우저용 주소(/data)로 치환
            if raw_path.startswith('/app/data'):
                path = raw_path.replace('/app/data', '/data')
            elif raw_path.startswith('data'):
                path = f"/data/{raw_path[5:]}" if raw_path.startswith('data/') else f"/{raw_path}"
            else:
                path = raw_path if raw_path.startswith('/') else f"/{raw_path}"
            
            # /data/data 처럼 중복되는 경우 방지
            if path.startswith('/data/data'):
                path = path.replace('/data/data', '/data')

            print(f"   [변환 주소] {file.file_type}: {path}")
            file_paths[file.file_type] = path

        print(f"{'='*50}\n")
        
        return {
            "success": True,
            "post_idx": post_idx,
            "total_score": post.total_score,
            "files": {
                "kf1_image": file_paths.get("KF1"),
                "kf2_image": file_paths.get("KF2"),
                "kf3_image": file_paths.get("KF3"),
                "backswing_video": file_paths.get("BACKSWING"),
                "impact_video": file_paths.get("IMPACT")
            },
            "keyframes": {
                "kf1": analysis.kf1 if analysis else None,
                "kf2": analysis.kf2 if analysis else None,
                "kf3": analysis.kf3 if analysis else None
            },
            "scores": analysis.score_json if analysis else {}
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# 싱글톤 인스턴스
video_analysis_service = VideoAnalysisService()