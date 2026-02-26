"""
최종 수정본: 스윙 분석 서비스 (GolfAnalyzer 방식 통합)
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


class SwingService:
    """스윙 분석 서비스 (실시간 창구 및 데이터 누적 관리)"""
    
    _initialized = False

    def __init__(self):
        current_file_path = os.path.abspath(__file__) 
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        self.save_dir = os.path.join(project_root, "data", "realtime")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.preprocessor = Preprocessor()
        self.keyframe_detector = KeyframeDetector()
        self.pose_detector = PoseDetector()
        
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
            """상체 회전 및 자세 교정 중심 피드백"""
            if total_score >= 70:
                # [잘함] 가슴 열기 + 높은 타점
                return "완벽해요! 상체 회전과 타점 모두 전문가 수준입니다."
            elif total_score >= 40:
                # [보통] 상체는 열리나 팔 각도가 아쉬움
                return "보통이에요! 스윙 시 어깨를 조금만 더 부드럽게 써보세요."
            else:
                # [나쁨] 상체를 아예 안 열고 정면만 보는 경우
                return "상체가 굳어있어요! 어깨를 뒤로 충분히 열어주세요."

    # ========================================
    # 실시간 분석 통합 메서드 (1~3회차 공통)
    # ========================================
    
    async def analyze_realtime(
        self, 
        request: SwingAnalysisRequest, 
        db: Session,
        user_id: str
    ):
        """실시간 스윙 분석 전체 프로세스"""
        self._validate_request(request)
        
        # 1. Keypoints 추출
        # 수정 - 프론트에서 받은 keypoints 직접 사용
        keypoints_list = []
        if request.keypoints:
            for frame_id, kp in enumerate(request.keypoints):
                if kp:
                    kp['frame_id'] = frame_id
                    keypoints_list.append(kp)

        if not keypoints_list:
            raise ValueError("사람이 감지되지 않았습니다. 전신이 보이게 촬영해주세요.")
        # 2. 키프레임 감지
        kf1, kf2, kf3 = self.detect_keyframes(keypoints_list)
        
        # 3. post_id 결정
        if request.swing_num == 1:
            post_id = str(uuid.uuid4())
        else:
            post_id = request.post_id

        # 4. 저장 폴더 생성
        swing_dir = os.path.join(self.save_dir, post_id, f"swing{request.swing_num}")
        os.makedirs(swing_dir, exist_ok=True)

        # 5. base64 프레임 → original.mp4 생성
        video_path = os.path.join(swing_dir, "original.mp4")
        self._frames_to_video(request.frames, video_path)

        # 6. GolfAnalyzer 방식으로 점수 계산 + 이미지/영상 저장
        df = pd.DataFrame(keypoints_list)
        eval_result = self.score_calculator.evaluate_user(
            df,
            {'ready': kf1, 'backswing': kf2, 'impact': kf3},
            video_path,
            swing_dir
        )

        total_score = eval_result['total_score']
        quick_feedback = self.get_quick_feedback(total_score)

        # 7. 회차별 분기 처리
        if request.swing_num == 1:
            return await self._process_swing_1(
                request, db, user_id, post_id, kf1, kf2, kf3, eval_result, quick_feedback, swing_dir
            )
        else:
            return await self._process_swing_2_or_3(
                request, db, post_id, kf1, kf2, kf3, eval_result, quick_feedback, swing_dir
            )

    # ========================================
    # 회차별 데이터베이스 처리 로직
    # ========================================

    async def _process_swing_1(
        self, 
        request, 
        db, 
        user_id,
        post_id,
        kf1, 
        kf2, 
        kf3, 
        eval_result, 
        quick_feedback,
        swing_dir
    ):
        """1회차 처리: 신규 기록 생성"""
        post = Post(
            idx=post_id,
            user_id=user_id,
            type="REALTIME",
            status="ANALYZING",
            total_score=eval_result['total_score']
        )
        db.add(post)
        db.flush()
        
        analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=post_id,
            swing_num=1,
            kf1=kf1,
            kf2=kf2,
            kf3=kf3,
            score_json={
                "details": eval_result['details'],
                "total_score": eval_result['total_score']
            }
        )
        db.add(analysis)
        
        self._register_swing_files(db, post_id, swing_dir, swing_num=1)
        
        db.commit()
        db.refresh(post)
        
        return QuickFeedbackResponse(
            swing_num=1,
            post_id=post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            total_score=eval_result['total_score'],
            stage_scores=self._calc_stage_scores(eval_result['details'])
        )

    async def _process_swing_2_or_3(
            self,
            request,
            db,
            post_id,
            kf1,
            kf2,
            kf3,
            eval_result,
            quick_feedback,
            swing_dir
        ):
            """2~3회차 처리: 평균 없이 개별 점수만 저장 및 반환"""
            post = db.query(Post).filter(Post.idx == post_id).first()
            if not post:
                raise ValueError("기존 분석 기록을 찾을 수 없습니다.")
            
            # 현재 스윙 점수 변수화
            current_score = eval_result['total_score']

            # 1. ANALYSIS 테이블에 개별 스윙 점수 저장
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                swing_num=request.swing_num,
                kf1=kf1,
                kf2=kf2,
                kf3=kf3,
                score_json={
                    "details": eval_result['details'],
                    "total_score": current_score # 🌟 해당 회차의 순수 점수
                }
            )
            db.add(analysis)
            
            self._register_swing_files(db, post_id, swing_dir, swing_num=request.swing_num)
            
            # 2. 3회차 완료 시 post의 대표 점수를 '마지막 스윙 점수' 혹은 '0'으로 설정
            # (평균을 안 쓰기로 했으므로, 마지막 스윙 점수로 업데이트하거나 유지합니다)
            if request.swing_num == 3:
                post.total_score = current_score # 마지막 스윙 점수를 대표 점수로
                post.status = "DONE"
                db.commit()
                
                files = db.query(File).filter(File.post_idx == post_id, File.swing_num == 3).all()
                file_paths = self._build_file_paths(files)
                
                return AnalysisCompleteResponse(
                    swing_num=3,
                    post_id=post_id,
                    save_to_db=True,
                    total_score=current_score, # 🌟 평균이 아닌 3회차 점수
                    stage_scores=self._calc_stage_scores(eval_result['details']),
                    quick_feedback=quick_feedback,
                    scores={
                        "details": eval_result['details'],
                        "total_score": current_score
                    },
                    keyframes={"kf1": kf1, "kf2": kf2, "kf3": kf3},
                    files=file_paths
                )
            
            db.commit()

            # 🌟 실시간 피드백 응답에도 해당 회차의 순수 점수만 보냄
            return QuickFeedbackResponse(
                swing_num=request.swing_num,
                post_id=post_id,
                quick_feedback=quick_feedback,
                save_to_db=True,
                total_score=current_score, # 🌟 꼬임 방지! 순수 현재 점수
                stage_scores=self._calc_stage_scores(eval_result['details'])
            )
        # ========================================
    # 유틸리티
    # ========================================

    def _validate_request(self, request):
        if request.swing_num < 1 or request.swing_num > 3:
            raise ValueError("swing_num 에러")
        if request.swing_num > 1 and not request.post_id:
            raise ValueError("post_id 누락")

    def _frames_to_video(self, frames: list, output_path: str, fps: int = 30):
        """base64 프레임 리스트 → mp4 영상 파일 생성"""
        temp_dir = tempfile.mkdtemp()
        try:
            for idx, frame_b64 in enumerate(frames):
                img_str = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64
                img_bytes = base64.b64decode(img_str)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{idx:04d}.jpg"), img)

            cmd = [
                'ffmpeg', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-y', output_path
            ]
            subprocess.run(cmd, capture_output=True, text=True)
        finally:
            shutil.rmtree(temp_dir)

    def _register_swing_files(self, db, post_id: str, swing_dir: str, swing_num: int):
        """GolfAnalyzer가 저장한 파일들을 DB에 등록"""
        files_map = [
            ("1_Ready.jpg",          "READY"),
            ("Seq_1_Ready.jpg",      "SEQ1_READY"),
            ("Seq_2_Takeaway.jpg",   "SEQ2_TAKEAWAY"),
            ("Seq_3_Backswing.jpg",  "SEQ3_BACKSWING"),
            ("Seq_4_Downswing_1.jpg","SEQ4_DOWNSWING1"),
            ("Seq_5_Downswing_2.jpg","SEQ5_DOWNSWING2"),
            ("Seq_6_Impact.jpg",     "SEQ6_IMPACT"),
            ("3_Impact.jpg",         "IMPACT"),
            ("4_FollowSwing.mp4",    "FOLLOWSWING"),
        ]

        for filename, file_type in files_map:
            filepath = os.path.join(swing_dir, filename)
            if not os.path.exists(filepath):
                print(f"⚠️ 파일 누락: {filename}")
                continue
            ext = filename.split(".")[-1]
            db.add(File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                swing_num=swing_num,
                file_type=file_type,
                file_name=filename,
                file_path=filepath.replace("\\", "/"),
                file_extension=ext,
                storage_type="LOCAL"
            ))

    def _build_file_paths(self, files) -> dict:
        mapping = {
            "READY":          "ready",        # ⭐ kf1_image → ready
            "SEQ1_READY":     "seq1_ready",
            "SEQ2_TAKEAWAY":  "seq2_takeaway",
            "SEQ3_BACKSWING": "seq3_backswing",
            "SEQ4_DOWNSWING1":"seq4_downswing1",
            "SEQ5_DOWNSWING2":"seq5_downswing2",
            "SEQ6_IMPACT":    "seq6_impact",
            "IMPACT":         "impact",       # ⭐ kf3_image → impact
            "FOLLOWSWING":    "followswing",  # ⭐ follow_video → followswing
        }
        result = {}
        for f in files:
            key = mapping.get(f.file_type)
            if key:
                clean = f.file_path.replace("\\", "/")
                for marker in ["backend/data/", "data/realtime/", "data/upload/"]:
                    idx = clean.find(marker)
                    if idx != -1:
                        result[key] = "/data/" + clean[idx + len(marker):]
                        break
                else:
                    result[key] = clean
        return result

    def _calc_stage_scores(self, details: dict) -> dict:
        """details → stage_scores 계산"""
        def avg(d):
            vals = [v.get('score', 0) for v in d.values() if isinstance(v, dict)]
            return round(sum(vals) / len(vals), 1) if vals else 0

        ready_s  = avg(details.get('Ready', {}))
        swing_s  = avg({**details.get('Rotation', {}), **details.get('Backswing', {})})
        impact_s = avg({**details.get('Impact', {}),
                        'follow': {'score': details.get('FollowSwing', {}).get('Performance', {}).get('score', 0)}})

        return {"stage1": ready_s, "stage2": swing_s, "stage3": impact_s}


swing_service = SwingService()