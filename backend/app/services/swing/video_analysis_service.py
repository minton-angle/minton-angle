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

from app.services.swing.engine.pose_detector import PoseDetector
from app.services.swing.engine.merged_keyframes import KeyframeDetector
from app.services.swing.engine.score_calculator import ScoreCalculator
# from app.services.swing.engine.analyze_single_user_overlay import OverlayGenerator


class VideoAnalysisService:
    """동영상 업로드 분석 서비스"""
    
    _initialized = False

    def __init__(self):
        current_file_path = os.path.abspath(__file__) 
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        
        self.upload_dir = os.path.join(project_root, "data", "upload")
        self.keyframe_dir = os.path.join(project_root, "data", "upload_keyframes")
        
        self.pose_detector = PoseDetector()
        self.keyframe_detector = KeyframeDetector()
        # self.overlay_generator = OverlayGenerator()
        
        gt_json_path = os.path.join(project_root, "data", "standard", "gt_evaluation.json")
        
        if os.path.exists(gt_json_path):
            self.score_calculator = ScoreCalculator(gt_json_path)
            
            # ⭐ 클래스 변수로 체크!
            if not VideoAnalysisService._initialized:
                print(f"✅ [VideoService] 전문가 기준 로드 성공: {gt_json_path}")
                VideoAnalysisService._initialized = True
        else:
            self.score_calculator = ScoreCalculator()
            if not VideoAnalysisService._initialized:
                print(f"⚠️ [VideoService] GT 파일 없음: {gt_json_path}")
                VideoAnalysisService._initialized = True
    
    async def analyze_video(self, user_id: str, video: UploadFile, db: Session):
        """영상 업로드 및 3단계 9개 항목 종합 분석 실행"""
        
        print(f"🔥 analyze_video 시작! user_id={user_id}")  # ⭐ 추가
        
        # 1. POST 생성
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
            
            # 2. 저장 폴더 생성 (상대 경로 기준)
            upload_path = os.path.join(self.upload_dir, post_id)
            os.makedirs(upload_path, exist_ok=True)
            
            # 3. 원본 영상 저장
            video_path = os.path.join(upload_path, "original.mp4")
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            print(f"✅ 영상 파일 저장 완료")
            
            # 4. MediaPipe 관절(Keypoint) 추출
            keypoints_list = self.pose_detector.extract_from_video(video_path)
            if not keypoints_list:
                raise ValueError("관절 좌표 추출 실패: 영상에 사람이 명확히 보이지 않습니다.")
            print(f"✅ 좌표 추출 완료: {len(keypoints_list)} 프레임")
            
            # 5. 데이터 가공 (DataFrame)
            df = pd.DataFrame(keypoints_list)
            
            # 6. 핵심 키프레임(E1, E2, E3) 감지
            keyframes = self.keyframe_detector.detect(df)
            
            # 키프레임 감지 실패 시 방어 로직
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
            
            # ⭐ 7. 통합 엔진을 이용한 9개 항목 점수 계산 (중복 로직 제거)
            # evaluate_user는 1/0 정수값을 반환하므로 JSON 직렬화 에러가 없습니다.
            # evaluation_result = self.score_calculator.evaluate_user(df, kf_indices)
            evaluation_result = self.score_calculator.evaluate_user(
                df, 
                kf_indices, 
                video_path,      # 원본 영상 경로
                upload_path      # 저장 폴더 (post_id 폴더)
            )

            total_score = evaluation_result['total_score']
            print(f"✅ 점수 산출 완료: {total_score}점")
            
            # 8. 오버레이 시각화 자료 생성 (이미지 3개 + 비디오 2개)
            # keyframe_folder = os.path.join(self.keyframe_dir, post_id)
            # os.makedirs(keyframe_folder, exist_ok=True)
            
            # # OverlayGenerator 호출 (기존 시각화 기능 유지)
            # self.overlay_generator.generate_all_outputs(
            #     video_path,
            #     kf_indices,
            #     keyframe_folder
            # )
            # print(f"✅ 전문가 비교 오버레이 파일 생성 완료")
            
            # 9. 생성된 파일 정보를 DB File 테이블에 등록
            # self._register_files_to_db(post_id, keyframe_folder, db)
            self._register_files_to_db(post_id, upload_path, db)

            # 10. ANALYSIS 데이터 저장 (상세 평가 데이터 포함)
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                kf1=kf_indices['ready'],
                kf2=kf_indices['backswing'],
                kf3=kf_indices['impact'],
                # score_json={
                #     "evaluation": evaluation_result['evaluation'],
                #     "stage_scores": evaluation_result['stage_scores'],
                #     "total_score": total_score
                # }
                # ✅ 변경 (details 구조로)
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
            
            # ✅ 변경
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
        
        # 파일명과 타입 매핑
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