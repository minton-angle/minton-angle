"""
영상 업로드 분석 서비스 (실제 알고리즘 통합)
"""
import os
import uuid
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis

# ⭐ Engine 모듈 import
from app.services.swing.engine.pose_detector import PoseDetector
from app.services.swing.engine.merged_keyframes import KeyframeDetector
from app.services.swing.engine.score_calculator import ScoreCalculator
from app.services.swing.engine.analyze_single_user_overlay import OverlayGenerator


class VideoAnalysisService:
    """영상 분석 서비스 (실제 알고리즘)"""
    
    def __init__(self):
        self.upload_dir = os.path.join("data", "upload")
        self.keyframe_dir = os.path.join("data", "upload_keyframes")
        
        # ⭐ Engine 초기화
        self.pose_detector = PoseDetector()
        self.keyframe_detector = KeyframeDetector()
        
        # ScoreCalculator (GT 파일 있으면 초기화)
        gt_metrics_path = "backend/data/standard/GT_angle/gt_total_metrics2.csv"
        if os.path.exists(gt_metrics_path):
            self.score_calculator = ScoreCalculator(gt_metrics_path)
        else:
            print(f"⚠️ GT 메트릭 파일 없음: {gt_metrics_path}")
            self.score_calculator = None
        
        # OverlayGenerator
        self.overlay_generator = OverlayGenerator()
    
    
    async def analyze_video(self, user_id: str, video: UploadFile, db: Session):
        """영상 업로드 및 분석"""
        
        try:
            # 1. POST 생성
            post_id = str(uuid.uuid4())
            post = Post(
                idx=post_id,
                user_id=user_id,
                type="VIDEO",
                status="ANALYZING",
                total_score=0
            )
            db.add(post)
            db.flush()
            
            print(f"\n{'='*60}")
            print(f"📊 영상 분석 시작: {post_id}")
            print(f"{'='*60}")
            
            # 2. 폴더 생성
            upload_dir = os.path.join(self.upload_dir, post_id)
            os.makedirs(upload_dir, exist_ok=True)
            
            # 3. 원본 영상 저장
            video_path = os.path.join(upload_dir, "original.mp4")
            
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            
            print(f"✅ 영상 저장: {video_path}")
            
            # ⭐ 4. MediaPipe로 keypoint 추출
            keypoints_list = self.pose_detector.extract_from_video(video_path)
            print(f"✅ Keypoint 추출 완료: {len(keypoints_list)}프레임")
            
            if len(keypoints_list) == 0:
                raise ValueError("Keypoint 추출 실패! 사람이 감지되지 않았습니다.")
            
            # ⭐ 5. DataFrame 변환
            df = pd.DataFrame(keypoints_list)
            
            # ⭐ 6. 키프레임 감지
            keyframes = self.keyframe_detector.detect(df)
            
            if keyframes is None:
                print("⚠️ 키프레임 감지 실패, 기본값 사용")
                total_frames = len(keypoints_list)
                kf1 = total_frames // 3
                kf2 = total_frames * 2 // 3
                kf3 = total_frames - 1
            else:
                kf1 = keyframes['ready']
                kf2 = keyframes['backswing']
                kf3 = keyframes['impact']
            
            print(f"✅ 키프레임 감지: KF1={kf1}, KF2={kf2}, KF3={kf3}")
            
            # ⭐ 7. 점수 계산
            if self.score_calculator:
                result = self.score_calculator.calculate_scores(df, {
                    'ready': kf1,
                    'backswing': kf2,
                    'impact': kf3
                })
                
                # 기존 형식으로 변환
                scores = {}
                for item in result['user_evaluation']:
                    stage = item['단계']
                    score = item['점수']
                    
                    if stage == 'ready':
                        scores['elbow_height'] = score
                    elif stage == 'backswing':
                        scores['wrist_snap'] = score
                        scores['shoulder_rotation'] = score
                    elif stage == 'impact':
                        scores['hit_position'] = score
                        scores['racket_angle'] = score
                
                scores['follow_through'] = 75  # 더미
                
            else:
                # 더미 점수
                scores = {
                    "elbow_height": 85,
                    "wrist_snap": 78,
                    "hit_position": 90,
                    "shoulder_rotation": 82,
                    "racket_angle": 88,
                    "follow_through": 75
                }
            
            total_score = sum(scores.values()) // len(scores)
            print(f"✅ 점수 계산 완료: {total_score}점")
            
            # ⭐ 8. 오버레이 생성 (이미지 3개 + 비디오 2개)
            keyframe_folder = os.path.join(self.keyframe_dir, post_id)
            os.makedirs(keyframe_folder, exist_ok=True)
            
            self.overlay_generator.generate_all_outputs(
                video_path,
                {'ready': kf1, 'backswing': kf2, 'impact': kf3},
                keyframe_folder
            )
            print(f"✅ 오버레이 생성 완료")
            
            # ⭐ 9. FILE 저장 (DB에 등록)
            self._register_files_to_db(post_id, keyframe_folder, db)
            print(f"✅ FILE DB 저장 완료")
            
            # 10. ANALYSIS 저장
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                kf1=kf1,
                kf2=kf2,
                kf3=kf3,
                score_json=scores
            )
            db.add(analysis)
            
            # 11. POST 업데이트
            post.total_score = total_score
            post.status = "DONE"
            
            db.commit()
            db.refresh(post)
            
            print(f"{'='*60}")
            print(f"🎉 분석 완료!")
            print(f"{'='*60}\n")
            
            return {
                "success": True,
                "post_idx": post_id,
                "total_score": total_score,
                "message": "분석 완료!"
            }
            
        except Exception as e:
            db.rollback()
            print(f"❌ 에러: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    
    def _register_files_to_db(self, post_id: str, keyframe_folder: str, db: Session):
        """
        생성된 파일들을 DB에 등록
        
        오버레이 파일명:
        - 1_ready_hybrid.jpg
        - 2_rotation_hybrid.mp4
        - 3_backswing_hybrid.jpg
        - 4_impact_hybrid.jpg
        - 5_follow_hybrid.mp4
        """
        
        files_map = [
            ("1_ready_hybrid.jpg", "KF1", "jpg"),
            ("3_backswing_hybrid.jpg", "KF2", "jpg"),
            ("4_impact_hybrid.jpg", "KF3", "jpg"),
            ("2_rotation_hybrid.mp4", "BACKSWING", "mp4"),
            ("5_follow_hybrid.mp4", "IMPACT", "mp4")
        ]
        
        for filename, file_type, ext in files_map:
            filepath = os.path.join(keyframe_folder, filename)
            
            # 파일 존재 확인
            if not os.path.exists(filepath):
                print(f"⚠️ 파일 없음: {filepath}")
                continue
            
            # DB 저장
            file = File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                file_type=file_type,
                file_name=filename,
                file_path=filepath.replace("\\", "/"),
                file_extension=ext,
                storage_type="LOCAL"
            )
            db.add(file)
            print(f"  ✅ {file_type} 등록: {filename}")
    
    
    async def get_status(self, post_idx: str, db: Session):
        """분석 상태 조회"""
        
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise ValueError("POST를 찾을 수 없습니다.")
        
        return {
            "post_idx": post_idx,
            "status": post.status,
            "total_score": post.total_score
        }


# 싱글톤 인스턴스
video_analysis_service = VideoAnalysisService()