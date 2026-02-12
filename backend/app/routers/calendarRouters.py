"""
캘린더 라우터
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db
from app.models.postModels import Post
from app.models.fileModels import File

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("/reports")
async def get_calendar_reports(
    user_id: str = Query(..., description="사용자 ID"),
    date: str = Query(..., description="날짜 (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    특정 날짜의 분석 리포트 조회
    - REALTIME 최신 1개
    - VIDEO 최신 1개
    """
    
    try:
        query_date = datetime.strptime(date, "%Y-%m-%d").date()
        start_datetime = datetime.combine(query_date, datetime.min.time())
        end_datetime = datetime.combine(query_date, datetime.max.time())
        
        print(f"\n{'='*50}")
        print(f"📅 캘린더 조회: {user_id} / {date}")
        
        # ⭐ REALTIME 최신 1개
        realtime_post = db.query(Post).filter(
            Post.user_id == user_id,
            Post.type == "REALTIME",
            Post.create_date >= start_datetime,
            Post.create_date <= end_datetime,
            Post.status == "DONE"
        ).order_by(Post.create_date.desc()).first()
        
        # ⭐ VIDEO 최신 1개
        video_post = db.query(Post).filter(
            Post.user_id == user_id,
            Post.type == "VIDEO",
            Post.create_date >= start_datetime,
            Post.create_date <= end_datetime,
            Post.status == "DONE"
        ).order_by(Post.create_date.desc()).first()
        
        print(f"REALTIME: {'있음' if realtime_post else '없음'}")
        print(f"VIDEO: {'있음' if video_post else '없음'}")
        
        # 리포트 데이터 생성
        def build_report(post):
            if not post:
                return None
            
            # 시간 포맷
            time_str = post.create_date.strftime("%H:%M")
            
            # 썸네일 (KF2 이미지)
            kf2_file = db.query(File).filter(
                File.post_idx == post.idx,
                File.file_type == "KF2"
            ).first()
            
            thumbnail = None
            if kf2_file:
                thumbnail = kf2_file.file_path if kf2_file.file_path.startswith('/') else f"/{kf2_file.file_path}"
            
            return {
                "post_idx": post.idx,
                "type": post.type,
                "time": time_str,
                "total_score": post.total_score,
                "thumbnail": thumbnail
            }
        
        realtime_report = build_report(realtime_post)
        video_report = build_report(video_post)
        
        print(f"{'='*50}\n")
        
        return {
            "success": True,
            "date": date,
            "realtime_report": realtime_report,  # ⭐ 실시간 리포트
            "video_report": video_report          # ⭐ 영상 리포트
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식이 잘못되었습니다. (YYYY-MM-DD)")
    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))