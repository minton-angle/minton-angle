"""
영상 업로드 분석 서비스
"""
import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis


class VideoAnalysisService:
    """영상 분석 서비스"""
    
    def __init__(self):
        self.upload_dir = os.path.join("data", "upload")
        self.keyframe_dir = os.path.join("data", "upload_keyframes")
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    
    async def analyze_video(self, user_id: str, video: UploadFile, db: Session):
        """영상 업로드 및 분석"""
        
        try:
            # 1. POST 생성 (UUID)
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
            
            print(f"\n{'='*50}")
            print(f"📊 영상 분석 시작: {post_id}")
            print(f"{'='*50}")
            
            # 2. 폴더명 = UUID
            folder_name = post_id
            
            # 3. 원본 영상 저장
            upload_dir = os.path.join(self.upload_dir, folder_name)
            os.makedirs(upload_dir, exist_ok=True)
            
            video_path = os.path.join(upload_dir, "original.mp4")
            
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            
            print(f"✅ 영상 저장: {video_path}")
            
            # 4. MediaPipe로 keypoint 추출
            keypoints = self._extract_keypoints(video_path)
            print(f"✅ Keypoint 추출 완료: {len(keypoints)}프레임")
            
            # 5. 키프레임 감지
            kf1, kf2, kf3 = self._detect_keyframes(keypoints)
            print(f"✅ 키프레임 감지: KF1={kf1}, KF2={kf2}, KF3={kf3}")
            
            # 6. 점수 계산
            scores = self._calculate_scores(keypoints, kf1, kf2, kf3)
            total_score = sum(scores.values()) // len(scores)
            print(f"✅ 점수 계산 완료: {total_score}점")
            
            # 7. 키프레임 저장 (이미지 3 + 동영상 2)
            await self._save_keyframes(folder_name, video_path, [kf1, kf2, kf3], db, post_id)
            print(f"✅ 키프레임 저장 완료")
            
            # 8. ANALYSIS 저장
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                kf1=kf1,
                kf2=kf2,
                kf3=kf3,
                score_json=scores
            )
            db.add(analysis)
            
            # 9. POST 업데이트
            post.total_score = total_score
            post.status = "DONE"
            
            db.commit()
            db.refresh(post)
            
            print(f"{'='*50}")
            print(f"🎉 분석 완료!")
            print(f"{'='*50}\n")
            
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
    
    
    def _extract_keypoints(self, video_path: str):
        """MediaPipe로 keypoint 추출"""
        
        cap = cv2.VideoCapture(video_path)
        keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 19개 keypoint 추출
                frame_kp = {
                    'nose_x': landmarks[0].x,
                    'nose_y': landmarks[0].y,
                    'right_shoulder_x': landmarks[12].x,
                    'right_shoulder_y': landmarks[12].y,
                    'right_elbow_x': landmarks[14].x,
                    'right_elbow_y': landmarks[14].y,
                    'right_wrist_x': landmarks[16].x,
                    'right_wrist_y': landmarks[16].y,
                }
                
                keypoints.append(frame_kp)
        
        cap.release()
        return keypoints
    
    
    def _detect_keyframes(self, keypoints):
        """키프레임 감지 (더미 버전)"""
        
        total_frames = len(keypoints)
        
        if total_frames < 3:
            return 0, 0, 0
        
        # 임시: 3등분
        kf1 = total_frames // 3
        kf2 = total_frames * 2 // 3
        kf3 = total_frames - 1
        
        return kf1, kf2, kf3
    
    
    def _calculate_scores(self, keypoints, kf1, kf2, kf3):
        """점수 계산 (더미 버전)"""
        
        return {
            "elbow_height": 85,
            "wrist_snap": 78,
            "hit_position": 90,
            "shoulder_rotation": 82,
            "racket_angle": 88,
            "follow_through": 75
        }
    
    
    async def _save_keyframes(self, folder_name, video_path, keyframe_indices, db, post_idx):
        """키프레임 저장 (이미지 3 + 동영상 2)"""
        
        kf1, kf2, kf3 = keyframe_indices
        
        # 키프레임 폴더 생성
        keyframe_folder = os.path.join(self.keyframe_dir, folder_name)
        os.makedirs(keyframe_folder, exist_ok=True)
        
        print(f"\n🎬 키프레임 저장 시작: {keyframe_folder}")
        
        # 1. 이미지 3개 저장
        cap = cv2.VideoCapture(video_path)
        
        for i, (kf_num, kf_idx) in enumerate([(1, kf1), (2, kf2), (3, kf3)], 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf_idx)
            ret, frame = cap.read()
            
            if ret:
                filename = f"kf{kf_num}.jpg"
                filepath = os.path.join(keyframe_folder, filename)
                cv2.imwrite(filepath, frame)
                
                # DB 저장
                file = File(
                    idx=str(uuid.uuid4()),
                    post_idx=post_idx,
                    file_type=f"KF{kf_num}",
                    file_name=filename,
                    file_path=filepath.replace("\\", "/"),
                    file_extension="jpg",
                    storage_type="LOCAL"
                )
                db.add(file)
                print(f"  ✅ KF{kf_num} 이미지 저장")
        
        cap.release()
        
        # 2. 백스윙 동영상 (KF1 → KF2)
        self._save_video_clip(
            video_path, keyframe_folder, kf1, kf2, "backswing.mp4", "BACKSWING", post_idx, db
        )
        print(f"  ✅ BACKSWING 동영상 저장")
        
        # 3. 임팩트 동영상 (KF2 → KF3)
        self._save_video_clip(
            video_path, keyframe_folder, kf2, kf3, "impact.mp4", "IMPACT", post_idx, db
        )
        print(f"  ✅ IMPACT 동영상 저장")
        
        print(f"🎉 키프레임 저장 완료\n")
    
    
    def _save_video_clip(self, video_path, output_dir, start_frame, end_frame, filename, file_type, post_idx, db):
        """동영상 클립 저장 (ffmpeg 사용 - 브라우저 호환)"""
        
        import subprocess
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 시작/끝 시간 계산
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        cap.release()
        
        # 출력 경로
        output_path = os.path.join(output_dir, filename)
        
        # ⭐ ffmpeg 명령어 (H.264 코덱, 브라우저 호환)
        cmd = [
            'ffmpeg',
            '-i', video_path,           # 입력 파일
            '-ss', str(start_time),     # 시작 시간
            '-t', str(duration),        # 지속 시간
            '-c:v', 'libx264',          # H.264 비디오 코덱
            '-preset', 'fast',          # 빠른 인코딩
            '-crf', '23',               # 품질 (18-28, 낮을수록 좋음)
            '-pix_fmt', 'yuv420p',      # 픽셀 포맷 (브라우저 호환)
            '-movflags', '+faststart',  # 웹 스트리밍 최적화
            '-y',                       # 덮어쓰기
            output_path
        ]
        
        print(f"  🎬 ffmpeg 실행: {filename}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ ffmpeg 에러:")
            print(result.stderr)
        else:
            print(f"  ✅ 동영상 생성 성공: {output_path}")
        
        # DB 저장
        file = File(
            idx=str(uuid.uuid4()),
            post_idx=post_idx,
            file_type=file_type,
            file_name=filename,
            file_path=output_path.replace("\\", "/"),
            file_extension="mp4",
            storage_type="LOCAL"
        )
        db.add(file)
    
    
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
video_analysis_service = VideoAnalysisService(),