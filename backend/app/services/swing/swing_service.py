"""
스윙 분석 서비스 (팀원 알고리즘 통합)
"""
import os
import uuid
import base64
import subprocess
import tempfile
import shutil
import pandas as pd
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

# ⭐ 팀원 알고리즘 Import
from .engine.gt_normalization_dtw import Preprocessor
from .engine.merged_keyframes import detect_keyframes_from_df
from .engine.score_calculator import GolfAnalyzer


class SwingService:
    """스윙 분석 서비스"""
    
    _initialized = False

    def __init__(self):
        current_file_path = os.path.abspath(__file__) 
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        self.save_dir = os.path.join(project_root, "data", "realtime")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ⭐ 팀원 엔진 초기화
        self.preprocessor = Preprocessor()
        
        if not SwingService._initialized:
            print(f"✅ [SwingService] 팀원 알고리즘 로드 완료")
            SwingService._initialized = True

    async def analyze_realtime(
        self, 
        request: SwingAnalysisRequest, 
        db: Session,
        user_id: str
    ):
        """실시간 스윙 분석"""
        
        self._validate_request(request)
        
        # 1. Base64 프레임 → 임시 영상 저장
        temp_video = self._frames_to_video(request.frames)
        
        # 2. 전처리 (Preprocessor)
        print(f"\n🔧 [SwingService] 전처리 시작...")
        df = self.preprocessor.process_video(temp_video)
        
        if df is None or df.empty:
            raise ValueError("키포인트 추출 실패")
        
        # 3. 키프레임 감지 (함수 직접 호출)
        print(f"\n🎯 [SwingService] 키프레임 감지 중...")
        keyframes = detect_keyframes_from_df(df)
        
        if keyframes is None:
            total_frames = len(df)
            kf1 = total_frames // 4
            kf2 = total_frames // 2
            kf3 = int(total_frames * 0.75)
        else:
            kf1 = int(keyframes['ready'])
            kf2 = int(keyframes['backswing'])
            kf3 = int(keyframes['impact'])
        
        # 4. 점수 계산 (GolfAnalyzer)
        print(f"\n📊 [SwingService] 점수 계산 중...")
        
        output_dir = os.path.join(self.save_dir, str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)
        
        temp_csv = os.path.join(output_dir, "keyframes.csv")
        keyframe_rows = [
            {'keyframe': 'ready', 'value': kf1},
            {'keyframe': 'backswing', 'value': kf2},
            {'keyframe': 'impact', 'value': kf3},
            {'keyframe': 'followswing', 'value': keyframes.get('followswing', 'X') if keyframes else 'X'}
        ]
        pd.DataFrame(keyframe_rows).to_csv(temp_csv, index=False)
        
        analyzer = GolfAnalyzer(temp_video, temp_csv, output_dir)
        analyzer.run()
        
        eval_result = analyzer.report
        
        # 5. 임시 파일 정리
        os.remove(temp_video)
        
        # 6. 빠른 피드백
        total_score = eval_result.get('total_score', 0)
        quick_feedback = self.get_quick_feedback(total_score)
        
        # 7. 회차별 분기
        if request.swing_num == 1:
            return await self._process_swing_1(
                request, db, user_id, kf1, kf2, kf3, 
                eval_result, quick_feedback, output_dir
            )
        else:
            return await self._process_swing_2_or_3(
                request, db, kf1, kf2, kf3, 
                eval_result, quick_feedback, output_dir
            )

    async def _process_swing_1(
        self, 
        request, 
        db, 
        user_id,
        kf1, 
        kf2, 
        kf3, 
        eval_result, 
        quick_feedback,
        temp_output_dir
    ):
        """1회차: 신규 POST 생성"""
        
        post_id = str(uuid.uuid4())
        
        post = Post(
            idx=post_id,
            user_id=user_id,
            type="REALTIME",
            status="ANALYZING",
            total_score=eval_result.get('total_score', 0)
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
            score_json=eval_result
        )
        db.add(analysis)
        
        self._move_files_to_post_dir(db, post_id, temp_output_dir, swing_num=1)
        
        db.commit()
        db.refresh(post)
        
        stage_scores = self._extract_stage_scores(eval_result)
        
        return QuickFeedbackResponse(
            swing_num=1,
            post_id=post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            total_score=eval_result.get('total_score', 0),
            stage_scores=stage_scores
        )

    async def _process_swing_2_or_3(
        self, 
        request, 
        db, 
        kf1, 
        kf2, 
        kf3, 
        eval_result, 
        quick_feedback,
        temp_output_dir
    ):
        """2~3회차: 기존 POST에 ANALYSIS 추가"""
        
        post = db.query(Post).filter(Post.idx == request.post_id).first()
        if not post:
            raise ValueError("기존 분석 기록을 찾을 수 없습니다.")
        
        analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=request.post_id,
            swing_num=request.swing_num,
            kf1=kf1,
            kf2=kf2,
            kf3=kf3,
            score_json=eval_result
        )
        db.add(analysis)
        
        self._move_files_to_post_dir(db, request.post_id, temp_output_dir, swing_num=request.swing_num)
        
        # 3회차 완료
        if request.swing_num == 3:
            all_analyses = db.query(Analysis).filter(Analysis.post_idx == request.post_id).all()
            avg_score = sum(a.score_json.get('total_score', 0) for a in all_analyses) // len(all_analyses)
            
            post.total_score = avg_score
            post.status = "DONE"
            db.commit()
            
            files = db.query(File).filter(
                File.post_idx == request.post_id,
                File.swing_num == 3
            ).all()
            
            file_paths = {}
            for f in files:
                clean_path = self._fix_path(f.file_path)
                
                if f.file_type == 'READY':
                    file_paths['kf1_image'] = clean_path
                elif f.file_type == 'FOLLOWSWING':
                    file_paths['impact_video'] = clean_path
            
            stage_scores = self._extract_stage_scores(eval_result)
            
            return AnalysisCompleteResponse(
                swing_num=3,
                post_id=request.post_id,
                save_to_db=True,
                total_score=avg_score,
                stage_scores=stage_scores,
                quick_feedback=quick_feedback,
                scores=eval_result,
                keyframes={
                    "kf1": kf1,
                    "kf2": kf2,
                    "kf3": kf3
                },
                files=file_paths
            )
        
        # 1~2회차
        db.commit()
        stage_scores = self._extract_stage_scores(eval_result)
        
        return QuickFeedbackResponse(
            swing_num=request.swing_num,
            post_id=request.post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            total_score=eval_result.get('total_score', 0),
            stage_scores=stage_scores
        )

    def _validate_request(self, request):
        if request.swing_num < 1 or request.swing_num > 3:
            raise ValueError("swing_num 에러")
        if request.swing_num > 1 and not request.post_id:
            raise ValueError("post_id 누락")

    def _frames_to_video(self, frames):
        """Base64 프레임들 → 임시 영상 파일"""
        
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, "temp.mp4")
        
        for idx, frame_base64 in enumerate(frames):
            img_str = frame_base64.split(",")[1] if "," in frame_base64 else frame_base64
            img_data = base64.b64decode(img_str)
            
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
            with open(frame_path, 'wb') as f:
                f.write(img_data)
        
        # ⭐ 브라우저 호환 코덱
        cmd = [
            'ffmpeg', '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y', temp_video
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return temp_video

    def _move_files_to_post_dir(self, db, post_id, temp_dir, swing_num):
        """임시 폴더 → 최종 저장 경로로 파일 이동 + DB 등록"""
        
        post_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(post_dir, exist_ok=True)
        
        # ⭐ GolfAnalyzer 결과 파일 매핑
        file_mapping = {
            '1_Ready.jpg': ('READY', 'jpg'),
            'Seq_1_Ready.jpg': ('SEQ1_READY', 'jpg'),
            'Seq_2_Takeaway.jpg': ('SEQ2_TAKEAWAY', 'jpg'),
            'Seq_3_Backswing.jpg': ('SEQ3_BACKSWING', 'jpg'),
            'Seq_4_Downswing_1.jpg': ('SEQ4_DOWNSWING1', 'jpg'),
            'Seq_5_Downswing_2.jpg': ('SEQ5_DOWNSWING2', 'jpg'),
            'Seq_6_Impact.jpg': ('SEQ6_IMPACT', 'jpg'),
            '3_Impact.jpg': ('IMPACT', 'jpg'),
            '4_FollowSwing.mp4': ('FOLLOWSWING', 'mp4')
        }
        
        for temp_name, (file_type, ext) in file_mapping.items():
            temp_path = os.path.join(temp_dir, temp_name)
            
            if os.path.exists(temp_path):
                final_name = f"swing{swing_num}_{file_type.lower()}.{ext}"
                final_path = os.path.join(post_dir, final_name)
                
                shutil.copy(temp_path, final_path)
                
                db.add(File(
                    idx=str(uuid.uuid4()),
                    post_idx=post_id,
                    swing_num=swing_num,
                    file_type=file_type,
                    file_name=final_name,
                    file_path=final_path.replace("\\", "/"),
                    file_extension=ext,
                    storage_type="LOCAL"
                ))
                
                print(f"✅ DB 등록: {file_type} → {final_name}")
            else:
                print(f"⚠️ 파일 없음: {temp_name}")

    def _extract_stage_scores(self, eval_result):
        """eval_result에서 stage_scores 추출"""
        
        details = eval_result.get('details', {})
        
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

    def _fix_path(self, raw_path):
        """절대 경로 → 웹 경로 변환"""
        if not raw_path:
            return ""
        
        clean_path = raw_path.replace("\\", "/")
        marker = "backend/data/"
        index = clean_path.find(marker)
        
        if index != -1:
            return "/" + clean_path[index:]
        
        return clean_path

    def get_quick_feedback(self, total_score):
        """총점 기반 빠른 피드백"""
        if total_score >= 90:
            return "완벽해요! 🎉"
        elif total_score >= 80:
            return "좋아요! 👍"
        elif total_score >= 70:
            return "괜찮아요! 💪"
        else:
            return "조금 더 연습해봐요! 📈"


swing_service = SwingService()