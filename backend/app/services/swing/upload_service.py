"""
Upload Service - 사용자 영상 분석

출력: JSON (LLM 담당자에게 전달)
"""

import json
import uuid
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, BinaryIO
import shutil

from .engine.pose_extractor import PoseExtractor, KeyFrames
from .engine.swing_analyzer import SwingAnalyzer
from .engine.constants import OUTPUT_FILES, PHASE1, PHASE2, PHASE3


# ============================================
# 경로 설정
# ============================================
def get_paths():
    current = Path(__file__).resolve().parent
    backend = current.parents[2]
    return {
        "uploads": backend / "uploads" / "swing",
        "outputs": backend / "outputs" / "swing",
        "criteria": backend / "data" / "swing_criteria.json",
    }


# ============================================
# Upload Service
# ============================================
class SwingUploadService:
    def __init__(self):
        self.paths = get_paths()
        self.extractor = PoseExtractor()
        self.criteria = self._load_criteria()
        self.analyzer = SwingAnalyzer(self.criteria)
    
    def _load_criteria(self) -> Optional[Dict]:
        path = self.paths["criteria"]
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _generate_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def save_upload(self, file: BinaryIO, filename: str) -> str:
        """업로드 파일 임시 저장"""
        self.paths["uploads"].mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix or '.mp4'
        temp_path = self.paths["uploads"] / f"temp_{uuid.uuid4().hex[:8]}{ext}"
        
        with open(temp_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        return str(temp_path)
    
    def analyze(self, video_path: str) -> Dict:
        """
        영상 분석 실행
        
        Returns:
            LLM에게 전달할 JSON
        """
        analysis_id = self._generate_id()
        output_dir = self.paths["outputs"] / analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 핵심 프레임 추출
            key_frames = self.extractor.extract_from_video(video_path)
            
            # 2. 분석
            result = self.analyzer.analyze(key_frames)
            
            # 3. 파일 저장
            files = self._save_files(video_path, key_frames, output_dir)
            
            # 4. 최종 JSON
            response = {
                "success": True,
                "analysis_id": analysis_id,
                "analyzed_at": datetime.now().isoformat(),
                
                # 파일 경로 (프론트엔드용)
                "files": {
                    "phase1_image": f"/api/swing/output/{analysis_id}/{OUTPUT_FILES['phase1_image']}",
                    "phase2_video": f"/api/swing/output/{analysis_id}/{OUTPUT_FILES['phase2_video']}",
                    "phase2_impact_image": f"/api/swing/output/{analysis_id}/{OUTPUT_FILES['phase2_impact_image']}",
                    "phase3_video": f"/api/swing/output/{analysis_id}/{OUTPUT_FILES['phase3_video']}",
                },
                
                # 분석 결과 (LLM용)
                "phases": result["phases"],
                "overall_score": result["overall_score"],
            }
            
            # 결과 저장
            with open(output_dir / "result.json", 'w') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            
            return response
            
        except Exception as e:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            return {"success": False, "error": str(e)}
        
        finally:
            # 임시 파일 삭제
            if "temp_" in video_path and Path(video_path).exists():
                Path(video_path).unlink()
    
    def _save_files(self, video_path: str, kf: KeyFrames, output_dir: Path) -> Dict:
        """결과 파일 저장"""
        files = {}
        
        # Phase 1: 준비 이미지
        p1_path = output_dir / OUTPUT_FILES["phase1_image"]
        cv2.imwrite(str(p1_path), kf.ready.image)
        files["phase1_image"] = str(p1_path)
        
        # Phase 2: 임팩트 이미지
        p2_img_path = output_dir / OUTPUT_FILES["phase2_impact_image"]
        cv2.imwrite(str(p2_img_path), kf.impact.image)
        files["phase2_impact_image"] = str(p2_img_path)
        
        # Phase 2: 백스윙→임팩트 영상
        p2_vid_path = output_dir / OUTPUT_FILES["phase2_video"]
        self._save_clip(video_path, kf.ready.frame_idx, kf.impact.frame_idx, str(p2_vid_path))
        files["phase2_video"] = str(p2_vid_path)
        
        # Phase 3: 팔로우스루 영상
        p3_vid_path = output_dir / OUTPUT_FILES["phase3_video"]
        self._save_clip(video_path, kf.impact.frame_idx, kf.end.frame_idx, str(p3_vid_path))
        files["phase3_video"] = str(p3_vid_path)
        
        return files
    
    def _save_clip(self, video_path: str, start: int, end: int, output_path: str):
        """영상 클립 저장"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for _ in range(end - start + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
    
    def get_result(self, analysis_id: str) -> Optional[Dict]:
        """저장된 결과 조회"""
        path = self.paths["outputs"] / analysis_id / "result.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def close(self):
        self.extractor.close()


# Singleton
_instance = None

def get_service() -> SwingUploadService:
    global _instance
    if _instance is None:
        _instance = SwingUploadService()
    return _instance