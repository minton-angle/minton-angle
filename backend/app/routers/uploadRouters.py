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
        
        # ⭐ 경로 변환 함수
        def fix_path(raw_path):
            if not raw_path:
                return ""
            clean_path = raw_path.replace("\\", "/")
            marker = "backend/data/"
            index = clean_path.find(marker)
            if index != -1:
                return "/" + clean_path[index:]
            return clean_path
        
        file_paths = {}
        for file in files:
            print(f"  {file.file_type}: {file.file_path}")
            clean_path = fix_path(file.file_path)
            file_paths[file.file_type] = clean_path
        
        print(f"변환된 경로:")
        for key, value in file_paths.items():
            print(f"  {key}: {value}")
        print(f"{'='*50}\n")
        
        # ⭐ stage_scores 추출
        def extract_stage_scores(score_json):
            details = score_json.get('details', {})
            
            def calc_phase_score(phase_name):
                phase_data = details.get(phase_name, {})
                if not phase_data:
                    return 0
                scores = []
                for key, value in phase_data.items():
                    if isinstance(value, dict) and 'score' in value:
                        scores.append(value['score'])
                return round(sum(scores) / len(scores), 2) if scores else 0
            
            return {
                'ready': calc_phase_score('Ready'),
                'rotation': calc_phase_score('Rotation'),
                'backswing': calc_phase_score('Backswing'),
                'impact': calc_phase_score('Impact'),
                'followswing': calc_phase_score('FollowSwing')
            }
        
        return {
            "success": True,
            "post_idx": post_idx,
            "type": post.type.lower(),
            "total_score": post.total_score,
            "files": {
                "kf1_image": file_paths.get("READY"),
                "seq1_ready": file_paths.get("SEQ1_READY"),
                "seq2_takeaway": file_paths.get("SEQ2_TAKEAWAY"),
                "seq3_backswing": file_paths.get("SEQ3_BACKSWING"),
                "seq4_downswing1": file_paths.get("SEQ4_DOWNSWING1"),
                "seq5_downswing2": file_paths.get("SEQ5_DOWNSWING2"),
                "seq6_impact": file_paths.get("SEQ6_IMPACT"),
                "kf3_image": file_paths.get("IMPACT"),
                "impact_video": file_paths.get("FOLLOWSWING")
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