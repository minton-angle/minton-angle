"""
영상 업로드 분석 서비스
"""
import os
import uuid
import cv2
import mediapipe as mp
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis
from app.services.swing.swing_service import swing_service


class VideoAnalysisService:
    """영상 분석 서비스"""
    
    def __init__(self):
        self.upload_dir = os.path.join("data", "upload")
        self.keyframe_dir = os.path.join("data", "upload_keyframes")
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    
    async def analyze_video(self, user_id: str, video: UploadFile, db: Session):
        """영상 업로드 및 분석 전체 프로세스"""
        
        # 파일명 추출 (확장자 제외)
        filename_without_ext = os.path.splitext(video.filename)[0]
        
        # 1. POST 생성 (UUID)
        post_idx = str(uuid.uuid4())
        post = Post(
            idx=post_idx,
            user_id=user_id,
            type="VIDEO",
            status="ANALYZING",
            total_score=0
        )
        db.add(post)
        db.commit()
        
        try:
            # 2. 영상 저장
            video_path = await self._save_video(filename_without_ext, video)
            
            # 3. Keypoints 추출
            keypoints_list = await self._extract_keypoints(video_path)
            
            # 4. 키프레임 감지
            kf1, kf2, kf3 = swing_service.detect_keyframes(keypoints_list)
            
            # 5. 점수 계산
            scores = swing_service.calculate_scores(keypoints_list, kf1, kf2, kf3)
            total_score = sum(scores.values()) // len(scores)
            
            # 6. 키프레임 저장 (이미지 3개 + 동영상 2개)
            await self._save_keyframes(
                folder_name=filename_without_ext,
                video_path=video_path,
                keyframe_indices=[kf1, kf2, kf3],
                db=db,
                post_idx=post_idx
            )
            
            # 7. ANALYSIS 저장
            analysis = Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_idx,
                kf1=kf1,
                kf2=kf2,
                kf3=kf3,
                score_json=scores
            )
            db.add(analysis)
            
            # 8. POST 업데이트
            post.status = "DONE"
            post.total_score = total_score
            db.commit()
            
            return {
                "post_idx": post_idx,
                "status": "DONE",
                "total_score": total_score,
                "message": "분석이 완료되었습니다!"
            }
            
        except Exception as e:
            # 에러 발생 시 POST 삭제
            db.delete(post)
            db.commit()
            raise e
    
    
    async def _save_video(self, folder_name: str, video: UploadFile) -> str:
        """영상 파일 저장"""
        
        # 폴더명 = 파일명
        post_dir = os.path.join(self.upload_dir, folder_name)
        os.makedirs(post_dir, exist_ok=True)
        
        # 원본 영상 저장
        ext = os.path.splitext(video.filename)[1]
        filename = f"original{ext}"
        filepath = os.path.join(post_dir, filename)
        
        with open(filepath, "wb") as f:
            content = await video.read()
            f.write(content)
        
        print(f"✅ 원본 영상 저장: {filepath}")
        return filepath
    
    
    async def _extract_keypoints(self, video_path: str) -> list:
        """MediaPipe로 keypoints 추출"""
        
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # 33개 keypoint 추출
                landmarks = results.pose_landmarks.landmark
                frame_keypoints = []
                
                for lm in landmarks:
                    frame_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                keypoints_list.append(frame_keypoints)
        
        cap.release()
        
        print(f"✅ Keypoints 추출 완료: {len(keypoints_list)}프레임")
        return keypoints_list
    
    
    async def _save_keyframes(
        self, folder_name: str, video_path: str, keyframe_indices: list, 
        db: Session, post_idx: str
    ):
        """
        키프레임 저장
        - 이미지 3개: KF1, KF2, KF3
        - 동영상 2개: BACKSWING (KF1→KF2), IMPACT (KF2→KF3)
        """
        
        kf1, kf2, kf3 = keyframe_indices
        
        print(f"\n🎬 키프레임 저장:")
        print(f"   📂 폴더: {folder_name}")
        print(f"   🆔 POST: {post_idx}")
        print(f"   📸 이미지 3개: KF1({kf1}), KF2({kf2}), KF3({kf3})")
        print(f"   🎥 동영상 2개: BACKSWING({kf1}→{kf2}), IMPACT({kf2}→{kf3})")
        
        # 1. 이미지 3개
        await self._save_single_image(folder_name, post_idx, video_path, kf1, 'KF1', db)
        await self._save_single_image(folder_name, post_idx, video_path, kf2, 'KF2', db)
        await self._save_single_image(folder_name, post_idx, video_path, kf3, 'KF3', db)
        
        # 2. 백스윙 동영상 (KF1 → KF2)
        await self._save_video_clip(
            folder_name, post_idx, video_path, kf1, kf2, 'BACKSWING', db
        )
        
        # 3. 임팩트 동영상 (KF2 → KF3)
        await self._save_video_clip(
            folder_name, post_idx, video_path, kf2, kf3, 'IMPACT', db
        )
        
        print(f"✅ 저장 완료: 이미지 3개 + 동영상 2개\n")
    
    
    async def _save_single_image(
        self, folder_name: str, post_idx: str, video_path: str, 
        frame_idx: int, file_type: str, db: Session
    ):
        """특정 프레임을 이미지로 저장"""
        
        cap = cv2.VideoCapture(video_path)
        
        # 폴더명 = 파일명
        kf_dir = os.path.join(self.keyframe_dir, folder_name)
        os.makedirs(kf_dir, exist_ok=True)
        
        # 해당 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 파일명: kf1.jpg, kf2.jpg, kf3.jpg
            filename = f"{file_type.lower()}.jpg"
            filepath = os.path.join(kf_dir, filename)
            cv2.imwrite(filepath, frame)
            
            # 파일 크기
            file_size = os.path.getsize(filepath)
            
            # DB 저장
            file = File(
                idx=str(uuid.uuid4()),
                post_idx=post_idx,
                file_type=file_type,  # 'KF1', 'KF2', 'KF3'
                file_name=filename,
                file_path=filepath.replace("\\", "/"),
                file_extension="jpg",
                file_size=file_size,
                storage_type="LOCAL"
            )
            db.add(file)
            
            print(f"  ✅ {file_type} 이미지: {filename}")
        
        cap.release()
    
    
    async def _save_video_clip(
        self, folder_name: str, post_idx: str, video_path: str, 
        start_frame: int, end_frame: int, file_type: str, db: Session
    ):
        """프레임 구간을 동영상으로 저장"""
        
        cap = cv2.VideoCapture(video_path)
        
        # 원본 영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 폴더명 = 파일명
        kf_dir = os.path.join(self.keyframe_dir, folder_name)
        os.makedirs(kf_dir, exist_ok=True)
        
        # 파일명: backswing.mp4, impact.mp4
        filename = f"{file_type.lower()}.mp4"
        filepath = os.path.join(kf_dir, filename)
        
        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        # 프레임 추출 및 저장
        frame_count = 0
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 구간 내 프레임만 저장
            if start_frame <= current_frame <= end_frame:
                out.write(frame)
                frame_count += 1
            
            # 구간 끝나면 종료
            if current_frame > end_frame:
                break
            
            current_frame += 1
        
        cap.release()
        out.release()
        
        # 파일 크기
        file_size = os.path.getsize(filepath)
        
        # DB 저장
        file = File(
            idx=str(uuid.uuid4()),
            post_idx=post_idx,
            file_type=file_type,  # 'BACKSWING', 'IMPACT'
            file_name=filename,
            file_path=filepath.replace("\\", "/"),
            file_extension="mp4",
            file_size=file_size,
            storage_type="LOCAL"
        )
        db.add(file)
        
        print(f"  ✅ {file_type} 동영상: {filename} ({frame_count}프레임)")
    
    
    async def get_status(self, post_idx: str, db: Session):
        """분석 상태 조회"""
        
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise ValueError("POST를 찾을 수 없습니다.")
        
        return {
            "post_idx": post_idx,
            "status": post.status,
            "progress": 100 if post.status == "DONE" else 50
        }
    
    
    async def get_result(self, post_idx: str, db: Session):
        """분석 결과 조회"""
        
        post = db.query(Post).filter(Post.idx == post_idx).first()
        
        if not post:
            raise ValueError("POST를 찾을 수 없습니다.")
        
        if post.status != "DONE":
            raise ValueError("분석이 완료되지 않았습니다.")
        
        # ANALYSIS 조회
        analysis = db.query(Analysis).filter(Analysis.post_idx == post_idx).first()
        
        # FILE 조회
        files = db.query(File).filter(File.post_idx == post_idx).all()
        
        # 이미지와 동영상 분류
        images = {}
        videos = {}
        
        for file in files:
            if file.file_extension == "jpg":
                images[file.file_type] = {
                    "path": file.file_path,
                    "size": file.file_size
                }
            elif file.file_extension == "mp4":
                videos[file.file_type] = {
                    "path": file.file_path,
                    "size": file.file_size
                }
        
        return {
            "post_idx": post_idx,
            "total_score": post.total_score,
            "scores": analysis.score_json if analysis else {},
            "images": images,  # KF1, KF2, KF3
            "videos": videos,  # BACKSWING, IMPACT
            "feedback": "분석 완료!"
        }


# 싱글톤
video_analysis_service = VideoAnalysisService()