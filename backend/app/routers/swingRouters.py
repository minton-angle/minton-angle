"""
swingRouters.py
===============
스윙 분석 API 엔드포인트
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from services.swing.gt_generator import get_gt_service 
from services.swing.upload_service import get_swing_service
from schemas.swingSchemas import (
    SwingAnalyzeResponse,
    SwingAnalyzeError,
    GTCreateResponse,
    CriteriaResponse,
    ExpertVideoResponse
)

router = APIRouter(prefix="/swing", tags=["swing"])


# ============================================================
# 영상 분석 API
# ============================================================

@router.post("/upload/analyze", response_model=SwingAnalyzeResponse)
async def analyze_swing_video(file: UploadFile = File(...)):
    """
    스윙 영상 분석
    
    - 영상 업로드 → 분석 → 결과 반환
    - 지원 포맷: mp4, mov, avi, webm
    - 최대 크기: 100MB
    
    Returns:
        SwingAnalyzeResponse: 분석 결과
    """
    # 파일 검증
    allowed_extensions = {'.mp4', '.mov', '.avi', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)})"
        )
    
    # 파일 크기 체크 (100MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 100MB 이하여야 합니다.")
    
    # 임시 저장
    service = get_swing_service()
    temp_path = service.upload_dir / f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 분석 실행
        result = service.analyze_video(str(temp_path))
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "분석 실패"))
        
        return result
        
    finally:
        # 임시 파일 삭제
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# 결과 파일 조회 API
# ============================================================

@router.get("/output/{analysis_id}/{filename}")
async def get_output_file(analysis_id: str, filename: str):
    """
    분석 결과 파일 조회 (이미지/영상)
    
    Args:
        analysis_id: 분석 ID
        filename: 파일명 (phase1_ready.jpg, phase2_backswing_impact.mp4 등)
    """
    service = get_swing_service()
    file_path = service.output_dir / analysis_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    # MIME 타입 결정
    ext = file_path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
    }
    
    media_type = media_types.get(ext, 'application/octet-stream')
    
    return FileResponse(file_path, media_type=media_type)


# ============================================================
# 전문가 영상 API
# ============================================================

@router.get("/expert-video", response_model=ExpertVideoResponse)
async def get_expert_video():
    """
    비교용 전문가 영상 URL 반환
    """
    # TODO: 실제 전문가 영상 경로 설정
    return ExpertVideoResponse(
        url="/api/swing/static/expert_demo.mp4",
        type="local",
        available=True
    )


# ============================================================
# GT/기준값 관련 API
# ============================================================

@router.get("/criteria", response_model=CriteriaResponse)
async def get_criteria():
    """
    현재 판정 기준 조회
    """
    service = get_swing_service()
    criteria = service.get_criteria()
    return CriteriaResponse(criteria=criteria)


@router.post("/admin/create-gt", response_model=GTCreateResponse)
async def create_gt(video_paths: List[str]):
    """
    [관리자] GT 기준값 생성
    
    전문가 영상들에서 지표를 추출하여 판정 기준 생성
    
    Args:
        video_paths: 전문가 영상 경로 리스트
    """
    service = get_swing_service()
    result = service.create_gt_from_videos(video_paths)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/admin/create-gt-upload")
async def create_gt_from_uploads(files: List[UploadFile] = File(...)):
    """
    [관리자] 업로드된 영상들로 GT 생성
    """
    service = get_swing_service()
    
    # 임시 저장
    temp_paths = []
    gt_dir = service.upload_dir / "gt_temp"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for file in files:
            temp_path = gt_dir / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_paths.append(str(temp_path))
        
        # GT 생성
        result = service.create_gt_from_videos(temp_paths)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    finally:
        # 임시 파일 삭제
        for path in temp_paths:
            if Path(path).exists():
                Path(path).unlink()
