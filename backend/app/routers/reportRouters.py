from typing import Any, Dict, Optional
import logging
import time

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.report.LLM_report import generate_report

# --- DB/ORM imports for post_idx-based report ---
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.postModels import Post
from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport

from app.models.fileModels import File


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(prefix="/api/report", tags=["Report"])


logger_api = logging.getLogger("app.api")

class PostureReportRequest(BaseModel):
    angles: Dict[str, float] = Field(..., description="관절별 오차각도")
    meta: Optional[Dict[str, Any]] = Field(default=None)
    lang: str = Field(default="ko")


class PostureReportResponse(BaseModel):
    report: Dict[str, Any]

def _mean_abs_kf_error(a: Analysis) -> float:
    vals = []
    for v in (a.kf1_error, a.kf2_error, a.kf3_error):
        try:
            if v is None:
                continue
            vals.append(abs(float(v)))
        except Exception:
            continue
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

# --------------- POST /api/report/posture ---------------
@router.post("/posture", response_model=PostureReportResponse)
def posture_report(payload: PostureReportRequest):
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/posture start lang=%s meta=%s", payload.lang, (payload.meta or {}))

    try:
        report = generate_report(
            angles=payload.angles,
            meta=payload.meta,
            lang=payload.lang,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.info(
            "POST /api/report/posture ok time_ms=%.1f severity=%s summary=%s",
            dt_ms,
            report.get("overall_severity"),
            (report.get("summary") or "")[:120],
        )
        return {"report": report}
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/posture failed time_ms=%.1f err=%s", dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))



# --------------- GET /api/report/analysis/post/{post_idx} ---------------
@router.get("/analysis/post/{post_idx}")
def get_analysis_by_post_alias(post_idx: str, db: Session = Depends(get_db)):
    """
    프론트 차트/테이블 렌더링용 API.
    post_idx 기준으로 Analysis 히스토리를 조회하여
    세션 목록(JSON)으로 반환합니다.
    """
    try:
        analyses = (
            db.query(Analysis)
            .filter(Analysis.post_idx == post_idx)
            .order_by(Analysis.create_date.asc())
            .all()
        )

        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx")

        sessions = []
        for a in analyses:
            mean_err = _mean_abs_kf_error(a)
            score = round(max(0, min(100, 100 - (mean_err / 20) * 100)))

            sessions.append({
                "idx": a.idx,
                "created_at": a.create_date.isoformat() if a.create_date else None,
                "frame": "ALL",  # 종합 페이지 기준
                "score": score,
                "kf_error": round(mean_err, 4),
            })

        logger_api.info(
            "[GET ANALYSIS] post_idx=%s row_count=%d",
            post_idx,
            len(sessions),
        )

        return {
            "post_idx": post_idx,
            "sessions": sessions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger_api.exception("[GET ANALYSIS] failed post_idx=%s err=%s", post_idx, str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --------------- POST /api/report/post/{post_idx} ---------------

class PostReportResponse(BaseModel):
    report: Dict[str, Any]


@router.post("/post/{post_idx}", response_model=PostReportResponse)
def posture_report_from_post(post_idx: str, lang: str = "ko", db: Session = Depends(get_db)):
    """
    post_idx 기준으로 Analysis 히스토리(여러 건)를 DB에서 조회한 뒤,
    - 최신 1건(latest) + 5일치(history) 추세(개선/악화)를 meta에 포함하여
    - kf1_error/kf2_error/kf3_error 기반 LLM 리포트를 생성합니다.
    """
    t0 = time.perf_counter()
    logger_api.info("POST /api/report/post/%s start lang=%s", post_idx, lang)

    try:
        analyses = (
            db.query(Analysis)
            .filter(Analysis.post_idx == post_idx)
            .order_by(Analysis.create_date.asc())
            .all()
        )
        if not analyses:
            raise HTTPException(status_code=404, detail="No analysis rows for this post_idx")

        logger_api.info(
            "[DB FETCH] post_idx=%s row_count=%d first_date=%s last_date=%s",
            post_idx,
            len(analyses),
            analyses[0].create_date.isoformat() if analyses[0].create_date else None,
            analyses[-1].create_date.isoformat() if analyses[-1].create_date else None,
        )

        latest = analyses[-1]
        first = analyses[0]

        # 최신값을 angles로 구성 (LLM 입력)
        angles = {
            "kf1_error": float(latest.kf1_error or 0.0),
            "kf2_error": float(latest.kf2_error or 0.0),
            "kf3_error": float(latest.kf3_error or 0.0),
        }

        # history(전체) + trend(처음 vs 최신) 구성
        first_mean = _mean_abs_kf_error(first)
        last_mean = _mean_abs_kf_error(latest)
        delta = last_mean - first_mean  # 음수면 개선(오차 감소), 양수면 악화
        trend = "improved" if delta < -1e-9 else ("worsened" if delta > 1e-9 else "flat")

        history = [
            {
                "idx": a.idx,
                "created_at": a.create_date.isoformat() if a.create_date else None,
                "kf1_error": float(a.kf1_error or 0.0),
                "kf2_error": float(a.kf2_error or 0.0),
                "kf3_error": float(a.kf3_error or 0.0),
                "mean_abs_kf_error": round(_mean_abs_kf_error(a), 4),
                "score_json": a.score_json or {},
            }
            for a in analyses
        ]

        meta: Dict[str, Any] = {
            "post_idx": post_idx,
            "latest": {
                "idx": latest.idx,
                "created_at": latest.create_date.isoformat() if latest.create_date else None,
                "kf1": latest.kf1,
                "kf2": latest.kf2,
                "kf3": latest.kf3,
                "score_json": latest.score_json or {},
            },
            "history": history,
            "trend": {
                "first_at": first.create_date.isoformat() if first.create_date else None,
                "last_at": latest.create_date.isoformat() if latest.create_date else None,
                "first_mean_abs_kf_error": round(first_mean, 4),
                "last_mean_abs_kf_error": round(last_mean, 4),
                "delta_mean_abs_kf_error": round(delta, 4),
                "direction": trend,  # improved | worsened | flat
            },
        }

        logger_api.info(
            "[LLM INPUT] post_idx=%s angles=%s trend=%s",
            post_idx,
            angles,
            meta.get("trend"),
        )
        # (2) LLM 호출
        report = generate_report(angles=angles, meta=meta, lang=lang)
        # (3) ✅ DB 저장 (llm_report)
        llm_row = LLMReport(
            idx=str(uuid.uuid4()),
            post_idx=post_idx,
            feedback=report,                 # ← JSON 그대로 저장
            create_date=datetime.utcnow(),
        )
        db.add(llm_row)
        db.commit()

        logger_api.info("[LLM SAVE] post_idx=%s llm_report_idx=%s", post_idx, llm_row.idx)

        # (4) 응답
        return {"report": report, "llm_report_idx": llm_row.idx}

    except HTTPException:
        raise
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger_api.exception("POST /api/report/post/%s failed time_ms=%.1f err=%s", post_idx, dt_ms, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
# --------------- GET /api/upload/result/{post_idx} ---------------
@router.get("/upload/result/{post_idx}")
async def get_upload_result(post_idx: str, db: Session = Depends(get_db)):
    """동영상 업로드 결과 조회"""
    
    try:
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
        
        analysis = db.query(Analysis).filter(Analysis.post_idx == post_idx).first()
        files = db.query(FileModel).filter(FileModel.post_idx == post_idx).all()
        
        print(f"\n{'='*50}")
        print(f"📊 결과 조회: {post_idx}")
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
            
            # ⭐ 경로 변환 적용
            clean_path = fix_path(file.file_path)
            file_paths[file.file_type] = clean_path
        
        print(f"변환된 경로:")
        for key, value in file_paths.items():
            print(f"  {key}: {value}")
        print(f"{'='*50}\n")
        
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
            "stage_scores": _extract_stage_scores(analysis.score_json) if analysis else {},
            "evaluation": []
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ⭐ stage_scores 추출 함수 추가
def _extract_stage_scores(score_json):
    """eval_result에서 stage_scores 추출"""
    
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

@router.get("/realtime/result/{post_idx}")
async def get_realtime_result(post_idx: str, db: Session = Depends(get_db)):
    """실시간 분석 결과 조회 (3회 스윙)"""
    
    try:
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
        
        # 모든 스윙 조회 (1, 2, 3회차)
        analyses = db.query(Analysis).filter(
            Analysis.post_idx == post_idx
        ).order_by(Analysis.swing_num).all()
        
        print(f"\n{'='*50}")
        print(f"📊 실시간 분석 결과 조회: {post_idx}")
        print(f"스윙 개수: {len(analyses)}")
        
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
        
        # 스윙별 데이터 구성
        swings = {}
        
        for analysis in analyses:
            swing_num = analysis.swing_num
            
            # 해당 스윙의 파일들 조회
            swing_files = db.query(File).filter(
                File.post_idx == post_idx,
                File.swing_num == swing_num
            ).all()
            
            print(f"\n스윙 {swing_num}회차: 파일 {len(swing_files)}개")
            
            file_paths = {}
            for f in swing_files:
                print(f"  {f.file_type}: {f.file_path}")
                clean_path = fix_path(f.file_path)
                file_paths[f.file_type] = clean_path
            
            # stage_scores 추출
            stage_scores = _extract_stage_scores(analysis.score_json)
            
            swings[str(swing_num)] = {
                "swing_num": swing_num,
                "total_score": analysis.score_json.get('total_score', 0),
                "stage_scores": stage_scores,
                "evaluation": [],
                "keyframes": {
                    "kf1": analysis.kf1,
                    "kf2": analysis.kf2,
                    "kf3": analysis.kf3
                },
                "scores": analysis.score_json,
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
                }
            }
        
        print(f"{'='*50}\n")
        
        return {
            "success": True,
            "post_idx": post_idx,
            "type": "realtime",
            "total_score": post.total_score,
            "swings": swings
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))