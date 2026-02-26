"""
영상 업로드 라우터
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.postModels import Post
from app.models.fileModels import File as FileModel
from app.models.analysisModels import Analysis
from app.services.swing.video_analysis_service import video_analysis_service

from app.routers.userRouters import get_current_user

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/video")
async def upload_video(
    current_user = Depends(get_current_user),
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """영상 업로드 및 분석"""
    
    try:
        user_id = current_user.id
        
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        
        with open(temp_video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        print(f"📹 임시 파일 저장: {temp_video_path}")
        
        result = await video_analysis_service.analyze_video(
            video_path=temp_video_path,
            db=db,
            user_id=user_id
        )
        
        print(f"📡 프론트엔드로 보낼 최종 데이터: {result}")
        
        os.remove(temp_video_path)
        os.rmdir(temp_dir)
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{post_idx}")
async def get_upload_status(post_idx: str, db: Session = Depends(get_db)):
    """분석 상태 조회"""
    
    try:
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")
        
        return {
            "post_idx": post_idx,
            "status": post.status,
            "total_score": post.total_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{post_idx}")
async def get_upload_result(post_idx: str, db: Session = Depends(get_db)):
    """동영상 업로드 분석 결과 조회"""
    
    try:
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
        
        analysis = db.query(Analysis).filter(Analysis.post_idx == post_idx).first()
        files = db.query(FileModel).filter(FileModel.post_idx == post_idx).all()
        
        print(f"\n{'='*50}")
        print(f"📊 업로드 결과 조회: {post_idx}")
        print(f"파일 개수: {len(files)}")
        
    #    # 파일 타입별 경로 매핑 - 도커 경로
    #     file_paths = {}
    #     for file in files:
    #         # DB에 저장된 원래 경로 (예: /app/data/upload_keyframes/...)
    #         raw_path = file.file_path
            
    #         print(f"   [원본 경로] {file.file_type}: {raw_path}")

    #         # 도커 내부 절대경로(/app/data)를 브라우저용 주소(/data)로 치환
    #         if raw_path.startswith('/app/data'):
    #             path = raw_path.replace('/app/data', '/data')
    #         elif raw_path.startswith('data'):
    #             path = f"/data/{raw_path[5:]}" if raw_path.startswith('data/') else f"/{raw_path}"
    #         else:
    #             path = raw_path if raw_path.startswith('/') else f"/{raw_path}"
            
    #         # /data/data 처럼 중복되는 경우 방지
    #         if path.startswith('/data/data'):
    #             path = path.replace('/data/data', '/data')

    #         print(f"   [변환 주소] {file.file_type}: {path}")
    #         file_paths[file.file_type] = path



        #파일 타입별 경로 매핑 - 로컬 경로 지정
        file_paths = {}
        for file in files:
            raw_path = file.file_path
            print(f"   [원본 경로] {file.file_type}: {raw_path}")

            # 역슬래시 → 슬래시
            clean_path = raw_path.replace("\\", "/")

            # backend/data/ 기준으로 웹 경로 추출
            marker = "backend/data/"
            idx = clean_path.find(marker)
            if idx != -1:
                path = "/data/" + clean_path[idx + len(marker):]  # ✅ /data/로 변환
            elif clean_path.startswith("/app/data"):
                path = clean_path.replace("/app/data", "/data")
            elif clean_path.startswith("/"):
                path = clean_path
            else:
                path = "/" + clean_path

            print(f"   [변환 주소] {file.file_type}: {path}")
            file_paths[file.file_type] = path
        print(f"{'='*50}\n")
        
        return {
            "success": True,
            "post_idx": post_idx,
            "type": post.type.lower(),
            "total_score": post.total_score,
            "files": {
                "ready":   file_paths.get("READY"),
                "seq1_ready":  file_paths.get("SEQ1_READY"),
                "seq2_takeaway": file_paths.get("SEQ2_TAKEAWAY"),
                "seq3_backswing": file_paths.get("SEQ3_BACKSWING"),
                "seq4_downswing1": file_paths.get("SEQ4_DOWNSWING1"),
                "seq5_downswing2": file_paths.get("SEQ5_DOWNSWING2"),
                "seq6_impact": file_paths.get("SEQ6_IMPACT"),
                "impact":   file_paths.get("IMPACT"),
                "followswing": file_paths.get("FOLLOWSWING"),
            },
            "keyframes": {
                "kf1": analysis.kf1 if analysis else None,
                "kf2": analysis.kf2 if analysis else None,
                "kf3": analysis.kf3 if analysis else None
            },
            "scores": analysis.score_json if analysis else {},
            "stage_scores": extract_stage_scores(analysis.score_json) if analysis else {},
            "evaluation": []
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# 싱글톤 인스턴스
# video_analysis_service = VideoAnalysisService()