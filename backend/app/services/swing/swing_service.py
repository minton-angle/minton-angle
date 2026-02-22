"""
최종 수정본: 스윙 분석 서비스 (9개 항목 점수 체계 및 상대 경로 최적화)
"""
import os
import uuid
import base64
import cv2
import numpy as np
import pandas as pd
import subprocess
import tempfile
import shutil
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis
from app.schemas.swing import (
    SwingAnalysisRequest,
    QuickFeedbackResponse,
    AnalysisCompleteResponse,
    ScoreDetail
)

# ⭐ Engine 모듈 Import
from .engine.gt_normalization_dtw import Preprocessor
from .engine.merged_keyframes import KeyframeDetector
from .engine.pose_detector import PoseDetector
from .engine.score_calculator import ScoreCalculator
from .engine.analyze_single_user_overlay import OverlayGenerator


class SwingService:
    """스윙 분석 서비스 (실시간 창구 및 데이터 누적 관리)"""
    
    _initialized = False

    def __init__(self):
        # 1. 실행 파일 위치 기준 프로젝트 루트 자동 계산
        current_file_path = os.path.abspath(__file__) 
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        # 2. 저장 경로 설정
        self.save_dir = os.path.join(project_root, "data", "realtime")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 3. 엔진 초기화
        self.preprocessor = Preprocessor()
        self.keyframe_detector = KeyframeDetector()
        self.pose_detector = PoseDetector()
        self.overlay_generator = OverlayGenerator()
        
        # ⭐ 통합된 9개 항목 엔진 로드
        gt_json_path = os.path.join(project_root, "data", "standard", "gt_evaluation.json")
        if os.path.exists(gt_json_path):
            self.score_calculator = ScoreCalculator(gt_json_path)
            
            if not SwingService._initialized:
                print(f"✅ [SwingService] GT 기준 로드 성공")
                SwingService._initialized = True
        else:
            self.score_calculator = ScoreCalculator()
            if not SwingService._initialized:
                print(f"⚠️ [SwingService] GT 파일 없음, 기본 엔진 사용")
                SwingService._initialized = True

    # ========================================
    # 핵심 분석 메서드
    # ========================================
    
    def detect_keyframes(self, keypoints_list):
        """키프레임 감지 (E1, E2, E3 인덱스 추출)"""
        df = pd.DataFrame(keypoints_list)
        result = self.keyframe_detector.detect(df)
        
        if result is None:
            total_frames = len(keypoints_list)
            return total_frames // 4, total_frames // 2, int(total_frames * 0.75)
        
        return int(result['ready']), int(result['backswing']), int(result['impact'])

    def get_quick_feedback(self, total_score):
        """총점에 따른 실시간 피드백 문구"""
        if total_score >= 90: return "완벽해요! 🎉"
        elif total_score >= 80: return "좋아요! 👍"
        elif total_score >= 70: return "괜찮아요! 💪"
        else: return "조금 더 연습해봐요! 📈"

    # ========================================
    # 실시간 분석 통합 메서드 (1~3회차 공통)
    # ========================================
    
    async def analyze_realtime(
        self, 
        request: SwingAnalysisRequest, 
        db: Session,
        user_id: str  # ⭐ 파라미터 추가!
    ):
        """실시간 스윙 분석 전체 프로세스"""
        self._validate_request(request)
        
        # 1. Keypoints 추출
        keypoints_list = []
        for frame_id, frame_base64 in enumerate(request.frames):
            keypoints = self.pose_detector.extract_from_base64(frame_base64)
            if keypoints:
                keypoints['frame_id'] = frame_id
                keypoints_list.append(keypoints)
        
        if not keypoints_list:
            raise ValueError("사람이 감지되지 않았습니다. 전신이 보이게 촬영해주세요.")
        
        # 2. 키프레임 감지
        kf1, kf2, kf3 = self.detect_keyframes(keypoints_list)
        
        # 3. 9개 항목 통합 엔진으로 점수 산출
        df = pd.DataFrame(keypoints_list)
        eval_result = self.score_calculator.evaluate_user(
            df, 
            {'ready': kf1, 'backswing': kf2, 'impact': kf3}
        )
        
        total_score = eval_result['total_score']
        quick_feedback = self.get_quick_feedback(total_score)
        
        # 4. 회차별 분기 처리
        if request.swing_num == 1:
            return await self._process_swing_1(
                request, db, user_id, kf1, kf2, kf3, eval_result, quick_feedback
            )
        else:
            return await self._process_swing_2_or_3(
                request, db, kf1, kf2, kf3, eval_result, quick_feedback
            )

    # ========================================
    # 회차별 데이터베이스 처리 로직
    # ========================================

    async def _process_swing_1(
        self, 
        request, 
        db, 
        user_id,
        kf1, 
        kf2, 
        kf3, 
        eval_result, 
        quick_feedback
    ):
        """1회차 처리: 신규 기록 생성"""
        post_id = str(uuid.uuid4())
        
        post = Post(
            idx=post_id,
            user_id=user_id,
            type="REALTIME",
            status="ANALYZING",
            total_score=eval_result['total_score']
        )
        db.add(post)
        db.flush()
        
        # ⭐ ANALYSIS 생성 (swing_num 추가)
        analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=post_id,
            swing_num=1,  # ⭐ 추가!
            kf1=kf1,
            kf2=kf2,
            kf3=kf3,
            score_json=eval_result
        )
        db.add(analysis)
        
        self._save_keyframe_files(db, post_id, request.frames, [kf1, kf2, kf3], swing_num=1)
        
        db.commit()
        db.refresh(post)
        
        return QuickFeedbackResponse(
            swing_num=1,
            post_id=post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            stage_scores=eval_result['stage_scores']
        )

    async def _process_swing_2_or_3(self, request, db, kf1, kf2, kf3, eval_result, quick_feedback):
        """2~3회차 처리: 각 스윙을 개별 ANALYSIS로 저장"""
        post = db.query(Post).filter(Post.idx == request.post_id).first()
        if not post:
            raise ValueError("기존 분석 기록을 찾을 수 없습니다.")
        
        # ⭐ 새로운 ANALYSIS 생성 (각 회차마다!)
        analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=request.post_id,
            swing_num=request.swing_num,  # ⭐ 2 또는 3
            kf1=kf1,
            kf2=kf2,
            kf3=kf3,
            score_json=eval_result
        )
        db.add(analysis)
        
        # ⭐ 파일 저장 (swing_num 포함)
        self._save_keyframe_files(db, request.post_id, request.frames, [kf1, kf2, kf3], swing_num=request.swing_num)
        
        # ⭐ 3회차 완료 시
        if request.swing_num == 3:
            # POST의 total_score는 3회차 평균으로 업데이트
            all_analyses = db.query(Analysis).filter(Analysis.post_idx == request.post_id).all()
            avg_score = sum(a.score_json.get('total_score', 0) for a in all_analyses) // len(all_analyses)
            
            post.total_score = avg_score
            post.status = "DONE"
            db.commit()
            
            # ⭐ 3회차 파일만 반환 (또는 전체 반환)
            files = db.query(File).filter(
                File.post_idx == request.post_id,
                File.swing_num == 3  # ⭐ 3회차 파일만
            ).all()
            
            file_paths = {}
            for f in files:
                if f.file_type == "KF1":
                    file_paths['kf1_image'] = f.file_path
                elif f.file_type == "KF2":
                    file_paths['kf2_image'] = f.file_path
                elif f.file_type == "KF3":
                    file_paths['kf3_image'] = f.file_path
                elif f.file_type == "BACKSWING":
                    file_paths['backswing_video'] = f.file_path
                elif f.file_type == "IMPACT":
                    file_paths['impact_video'] = f.file_path
            
            return AnalysisCompleteResponse(
                swing_num=3,
                post_id=request.post_id,
                save_to_db=True,
                total_score=avg_score,
                stage_scores=eval_result['stage_scores'],
                quick_feedback=quick_feedback,
                scores=eval_result,
                keyframes={
                    "kf1": kf1,
                    "kf2": kf2,
                    "kf3": kf3
                },
                files=file_paths
            )
        
        # ⭐ 1~2회차는 기존 로직
        db.commit()
        return QuickFeedbackResponse(
            swing_num=request.swing_num,
            post_id=request.post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            stage_scores=eval_result['stage_scores']
        )

    # ========================================
    # 유틸리티 (기존 로직 유지)
    # ========================================

    def _validate_request(self, request):
        if request.swing_num < 1 or request.swing_num > 3:
            raise ValueError("swing_num 에러")
        if request.swing_num > 1 and not request.post_id:
            raise ValueError("post_id 누락")

    def _save_keyframe_files(self, db, post_id, frames, keyframe_indices, swing_num):
        kf1, kf2, kf3 = keyframe_indices
        for kf_num, kf_idx in enumerate([kf1, kf2, kf3], 1):
            file_path = self._save_image(post_id, kf_num, swing_num, frames[kf_idx])
            db.add(File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                swing_num=swing_num,
                file_type=f"KF{kf_num}",
                file_name=f"s{swing_num}_kf{kf_num}.jpg",
                file_path=file_path,
                file_extension="jpg",
                storage_type="LOCAL"
            ))
        
        bs_path = self._save_video_from_frames(post_id, swing_num, frames, kf1, kf2, 'BACKSWING')
        im_path = self._save_video_from_frames(post_id, swing_num, frames, kf2, kf3, 'IMPACT')
        
        for p, t in [(bs_path, "BACKSWING"), (im_path, "IMPACT")]:
            db.add(File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                swing_num=swing_num,
                file_type=t,
                file_name=f"s{swing_num}_{t.lower()}.mp4",
                file_path=p,
                file_extension="mp4",
                storage_type="LOCAL"
            ))

    def _save_image(self, post_id, kf_num, swing_num, image_base64):
        post_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(post_dir, exist_ok=True)
        img_str = image_base64.split(",")[1] if "," in image_base64 else image_base64
        filepath = os.path.join(post_dir, f"swing{swing_num}_kf{kf_num}.jpg")
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(img_str))
        return filepath.replace("\\", "/")

    def _save_video_from_frames(self, post_id, swing_num, frames, start_idx, end_idx, video_type):
        post_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(post_dir, exist_ok=True)
        filepath = os.path.join(post_dir, f"swing{swing_num}_{video_type.lower()}.mp4")
        temp_dir = tempfile.mkdtemp()
        try:
            for idx, i in enumerate(range(start_idx, end_idx + 1)):
                img_str = frames[i].split(",")[1] if "," in frames[i] else frames[i]
                with open(os.path.join(temp_dir, f"frame_{idx:04d}.jpg"), 'wb') as f:
                    f.write(base64.b64decode(img_str))
            cmd = [
                'ffmpeg', '-framerate', '30',
                '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-y', filepath
            ]
            subprocess.run(cmd, capture_output=True, text=True)
        finally:
            shutil.rmtree(temp_dir)
        return filepath.replace("\\", "/")


swing_service = SwingService()