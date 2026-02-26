"""
영상 업로드 분석 서비스 (GDR 방식 3단계 9개 평가 항목 적용 및 중복 제거 버전)
"""
import os
import uuid
import pandas as pd
import traceback
import subprocess  # 🌟 추가: FFmpeg 실행용
import shutil      # 🌟 추가: 파일 복사용
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis

from app.services.swing.engine.pose_detector import PoseDetector
from app.services.swing.engine.merged_keyframes import KeyframeDetector
from app.services.swing.engine.score_calculator import ScoreCalculator

class VideoAnalysisService:
    """영상 분석 서비스 - 컨트롤 타워 (GDR 스타일)"""
    
    _initialized = False    

    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        self.upload_dir = os.path.join(project_root, "data", "upload")
        self.keyframe_dir = os.path.join(project_root, "data", "upload_keyframes")
        
        self.pose_detector = PoseDetector()
        self.keyframe_detector = KeyframeDetector()
        
        gt_json_path = os.path.join(project_root, "data", "standard", "gt_evaluation.json")
        
        if os.path.exists(gt_json_path):
            self.score_calculator = ScoreCalculator(gt_json_path)
            if not VideoAnalysisService._initialized:
                print(f"✅ [VideoService] 전문가 기준 로드 성공: {gt_json_path}")
                VideoAnalysisService._initialized = True
        else:
            self.score_calculator = ScoreCalculator()
            if not VideoAnalysisService._initialized:
                print(f"⚠️ [VideoService] GT 파일 없음: {gt_json_path}")
                VideoAnalysisService._initialized = True
    
    async def analyze_video(self, user_id: str, video: UploadFile, db: Session):
        """영상 업로드 및 물리적 회전 보정 후 분석 실행"""
        
        print(f"🔥 analyze_video 시작! user_id={user_id}")
        
        post_id = str(uuid.uuid4())
        try:
            # 1. 초기 POST 생성 (상태: ANALYZING)
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
            print(f"📊 [GDR 스타일] 스윙 분석 공장 가동: {post_id}")
            print(f"{'='*60}")
            
            # 2. 저장 폴더 생성
            upload_path = os.path.join(self.upload_dir, post_id)
            os.makedirs(upload_path, exist_ok=True)
            
            # 3. 🌟 원본 영상 임시 저장 및 물리적 회전 보정
            temp_raw_path = os.path.join(upload_path, "temp_raw.mp4")
            video_path = os.path.join(upload_path, "original.mp4") # 최종 분석용 경로
            
            # 일단 업로드된 파일을 임시로 저장
            with open(temp_raw_path, "wb") as f:
                content = await video.read()
                f.write(content)

            print(f"🔄 영상 회전 보정(FFmpeg) 시작...")
            try:
                # 💡 FFmpeg를 사용하여 메타데이터 회전값을 픽셀에 물리적으로 적용
                ffmpeg_cmd = [
                    "ffmpeg", "-y", 
                    "-display_rotation", "0", 
                    "-i", temp_raw_path,
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", # 해상도 짝수 맞춤
                    "-metadata:s:v", "rotate=0",           # 회전 꼬리표 초기화
                    "-c:v", "libx264", "-preset", "ultrafast", 
                    "-c:a", "copy", 
                    video_path
                ]
                # subprocess 실행 시 들여쓰기 주의
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"✅ 영상 물리적 회전 보정 완료")
                
                if os.path.exists(temp_raw_path):
                    os.remove(temp_raw_path)
            except Exception as ffmpeg_err:
                print(f"⚠️ FFmpeg 보정 실패 (원본 사용): {ffmpeg_err}")
                if os.path.exists(temp_raw_path):
                    os.rename(temp_raw_path, video_path)

            # 4. MediaPipe 관절(Keypoint) 추출
            keypoints_list = self.pose_detector.extract_from_video(video_path)
            if not keypoints_list:
                raise ValueError("관절 좌표 추출 실패: 영상에 사람이 명확히 보이지 않습니다.")
            print(f"✅ 좌표 추출 완료: {len(keypoints_list)} 프레임")
            
            # 5. 데이터 가공 (DataFrame)
            df = pd.DataFrame(keypoints_list)
            
            # 6. 핵심 키프레임(ready, backswing, impact) 감지
            keyframes = self.keyframe_detector.detect(df)
            
            if keyframes is None:
                total_f = len(keypoints_list)
                kf_indices = {'ready': total_f // 4, 'backswing': total_f // 2, 'impact': int(total_f * 0.75)}
            else:
                kf_indices = {
                    'ready': int(keyframes['ready']),
                    'backswing': int(keyframes['backswing']),
                    'impact': int(keyframes['impact'])
                }
            
            print(f"✅ 키프레임 확정: {kf_indices}")
            
            # 7. 통합 엔진을 이용한 9개 항목 점수 계산
            evaluation_result = self.score_calculator.evaluate_user(
                df, 
                kf_indices, 
                video_path,      # 보정된 영상 경로
                upload_path      # 저장 폴더
            )

            total_score = evaluation_result['total_score']
            print(f"✅ 점수 산출 완료: {total_score}점")
            
            # 9. 생성된 파일 정보를 DB File 테이블에 등록
            self._register_files_to_db(post_id, upload_path, db)

            # 10. ANALYSIS 데이터 저장
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                kf1=kf_indices['ready'],
                kf2=kf_indices['backswing'],
                kf3=kf_indices['impact'],
                score_json={
                    "details": evaluation_result['details'],
                    "total_score": total_score
                }
            )
            db.add(analysis)
            
            # 11. 최종 POST 상태 업데이트 및 커밋
            post.total_score = total_score
            post.status = "DONE"
            
            db.commit()
            db.refresh(post)
            
            print(f"{'='*60}")
            print(f"🎉 모든 분석 공정 완료! 최종 점수: {total_score}")
            print(f"{'='*60}\n")
            
            return {
                "success": True,
                "post_idx": post_id,
                "total_score": total_score,
                "details": evaluation_result.get('details', {}),
                "message": "분석이 성공적으로 완료되었습니다."
            }
            
        except Exception as e:
            db.rollback()
            try:
                fail_post = db.query(Post).filter(Post.idx == post_id).first()
                if fail_post:
                    fail_post.status = "FAILED"
                    db.commit()
            except:
                pass
                
            print(f"❌ 분석 실패: {str(e)}")
            traceback.print_exc()
            raise e
    
    def _register_files_to_db(self, post_id: str, keyframe_folder: str, db: Session):
        """생성된 이미지 및 비디오 결과 파일들을 DB에 일괄 등록"""
        files_map = [
            ("1_Ready.jpg", "READY", "jpg"),
            ("Seq_1_Ready.jpg", "SEQ1_READY", "jpg"),
            ("Seq_2_Takeaway.jpg", "SEQ2_TAKEAWAY", "jpg"),
            ("Seq_3_Backswing.jpg", "SEQ3_BACKSWING", "jpg"),
            ("Seq_4_Downswing_1.jpg", "SEQ4_DOWNSWING1", "jpg"),
            ("Seq_5_Downswing_2.jpg", "SEQ5_DOWNSWING2", "jpg"),
            ("Seq_6_Impact.jpg", "SEQ6_IMPACT", "jpg"),
            ("3_Impact.jpg", "IMPACT", "jpg"),
            ("4_FollowSwing.mp4", "FOLLOWSWING", "mp4"),
        ]
        
        for filename, file_type, ext in files_map:
            filepath = os.path.join(keyframe_folder, filename)
            if not os.path.exists(filepath):
                continue
            
            web_path = filepath.replace("\\", "/")
            
            file_entry = File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                file_type=file_type,
                file_name=filename,
                file_path=web_path,
                file_extension=ext,
                storage_type="LOCAL"
            )
            db.add(file_entry)
    
    async def get_status(self, post_idx: str, db: Session):
        """분석 상태 조회"""
        post = db.query(Post).filter(Post.idx == post_idx).first()
        if not post:
            raise ValueError("해당 분석 기록을 찾을 수 없습니다.")
        
        return {
            "post_idx": post_idx,
            "status": post.status,
            "total_score": post.total_score
        }

# 싱글톤 인스턴스 생성
video_analysis_service = VideoAnalysisService()