"""
동영상 업로드 분석 서비스 (팀원 알고리즘 통합)
"""
import os
import uuid
import shutil
import pandas as pd
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis

# ⭐ 팀원 알고리즘 Import
from .engine.gt_normalization_dtw import Preprocessor
from .engine.merged_keyframes import detect_keyframes_from_df
from .engine.score_calculator import GolfAnalyzer


class VideoAnalysisService:
    """동영상 업로드 분석 서비스"""
    
    _initialized = False

    def __init__(self):
        current_file_path = os.path.abspath(__file__) 
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        self.save_dir = os.path.join(project_root, "data", "upload")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ⭐ 팀원 엔진 초기화
        self.preprocessor = Preprocessor()
        
        if not VideoAnalysisService._initialized:
            print(f"✅ [VideoAnalysisService] 팀원 알고리즘 로드 완료")
            VideoAnalysisService._initialized = True

    async def analyze_video(self, video_path: str, db: Session, user_id: str):
        """동영상 업로드 분석"""
        
        # 1. POST 생성
        post_id = str(uuid.uuid4())
        post = Post(
            idx=post_id,
            user_id=user_id,
            type="VIDEO",
            status="ANALYZING"
        )
        db.add(post)
        db.flush()
        
        # 2. 전처리
        print(f"\n🔧 [VideoAnalysis] 전처리 시작...")
        df = self.preprocessor.process_video(video_path)
        
        if df is None or df.empty:
            raise ValueError("키포인트 추출 실패")
        
        # 3. 키프레임 감지
        print(f"\n🎯 [VideoAnalysis] 키프레임 감지 중...")
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
        
        # 4. 점수 계산
        print(f"\n📊 [VideoAnalysis] 점수 계산 중...")
        
        output_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(output_dir, exist_ok=True)
        
        temp_csv = os.path.join(output_dir, "keyframes.csv")
        keyframe_rows = [
            {'keyframe': 'ready', 'value': kf1},
            {'keyframe': 'backswing', 'value': kf2},
            {'keyframe': 'impact', 'value': kf3},
            {'keyframe': 'followswing', 'value': keyframes.get('followswing', 'X') if keyframes else 'X'}
        ]
        pd.DataFrame(keyframe_rows).to_csv(temp_csv, index=False)
        
        analyzer = GolfAnalyzer(video_path, temp_csv, output_dir)
        analyzer.run()
        
        eval_result = analyzer.report
        
        # 5. ANALYSIS 저장
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
        
        # 6. 파일 DB 등록
        self._register_files(db, post_id, output_dir)
        
        # 7. POST 완료
        total_score = eval_result.get('total_score', 0)
        post.total_score = total_score
        post.status = "DONE"
        
        db.commit()
        db.refresh(post)
        
        print(f"\n✅ [VideoAnalysis] 분석 완료: post_id={post_id}, score={total_score}")
        
        return {
            "post_id": post_id,
            "total_score": total_score,
            "status": "DONE"
        }

    def _register_files(self, db, post_id, output_dir):
        """GolfAnalyzer 결과 파일 DB 등록"""
        
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
            file_path = os.path.join(output_dir, temp_name)
            
            if os.path.exists(file_path):
                db.add(File(
                    idx=str(uuid.uuid4()),
                    post_idx=post_id,
                    swing_num=1,
                    file_type=file_type,
                    file_name=temp_name,
                    file_path=file_path.replace("\\", "/"),
                    file_extension=ext,
                    storage_type="LOCAL"
                ))
                print(f"✅ DB 등록: {file_type} → {temp_name}")
            else:
                print(f"⚠️ 파일 없음: {temp_name}")


video_analysis_service = VideoAnalysisService()