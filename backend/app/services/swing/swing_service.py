"""
스윙 분석 서비스 (모든 비즈니스 로직)
"""
import os
import uuid
import base64
import cv2
import numpy as np
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


class SwingService:
    """스윙 분석 서비스"""
    
    def __init__(self):
        # ⭐ 저장 경로: data/realtime
        self.save_dir = os.path.join("data", "realtime")
    
    
    # ========================================
    # 기본 분석 메서드 (더미)
    # ========================================
    
    def detect_keyframes(self, keypoints):
        """키프레임 감지 (임시 더미)"""
        total_frames = len(keypoints)
        kf1 = total_frames // 3
        kf2 = total_frames * 2 // 3
        kf3 = total_frames - 1
        return kf1, kf2, kf3
    
    
    def calculate_scores(self, keypoints, kf1, kf2, kf3):
        """점수 계산 (임시 더미)"""
        return {
            "elbow_height": 85,
            "wrist_snap": 78,
            "hit_position": 90,
            "shoulder_rotation": 82,
            "racket_angle": 88,
            "follow_through": 75
        }
    
    
    def get_quick_feedback(self, scores):
        """빠른 피드백 생성"""
        avg = sum(scores.values()) / len(scores)
        if avg >= 90:
            return "완벽해요! 🎉"
        elif avg >= 80:
            return "좋아요! 👍"
        elif avg >= 70:
            return "괜찮아요! 💪"
        else:
            return "조금 더 연습해봐요! 📈"
    
    
    def generate_detailed_feedback(self, scores, kf1, kf2, kf3):
        """상세 피드백 생성 (3회차용)"""
        avg = sum(scores.values()) / len(scores)
        
        if avg >= 90:
            overall = "훌륭합니다! 거의 완벽한 스윙이에요! 🎉"
        elif avg >= 80:
            overall = "아주 좋아요! 조금만 더 보완하면 완벽! 👍"
        elif avg >= 70:
            overall = "괜찮아요! 몇 가지 개선 포인트가 있네요. 💪"
        else:
            overall = "기본기부터 다시 연습해봐요! 📈"
        
        return {
            "overall": overall,
            "details": scores,
            "strengths": ["팔꿈치 높이가 좋아요!", "타구 위치가 정확해요!"],
            "improvements": ["손목 스냅을 조금 더 활용해보세요."],
            "next_goals": ["손목 스냅 80점 → 85점", "전체 평균 90점 달성"]
        }
    
    
    # ========================================
    # 실시간 분석 통합 메서드
    # ========================================
    
    async def analyze_realtime(
        self,
        request: SwingAnalysisRequest,
        db: Session
    ):
        """실시간 스윙 분석 전체 프로세스"""
        
        # 1. 검증
        self._validate_request(request)
        
        # 2. 키프레임 감지 & 점수 계산
        kf1, kf2, kf3 = self.detect_keyframes(request.keypoints)
        scores = self.calculate_scores(request.keypoints, kf1, kf2, kf3)
        quick_feedback = self.get_quick_feedback(scores)
        total_score = sum(scores.values()) // len(scores)
        
        # 3. 회차별 처리
        if request.swing_num == 1:
            return await self._process_swing_1(
                request, db, kf1, kf2, kf3, scores, quick_feedback, total_score
            )
        else:
            return await self._process_swing_2_or_3(
                request, db, kf1, kf2, kf3, scores, quick_feedback, total_score
            )
    
    
    def _validate_request(self, request):
        """요청 검증"""
        if request.swing_num < 1 or request.swing_num > 3:
            raise ValueError("swing_num은 1, 2, 3만 가능합니다.")
        
        if request.swing_num > 1 and not request.post_id:
            raise ValueError(f"{request.swing_num}회차는 post_id가 필수입니다. 1회차부터 시작하세요.")
        
        if request.swing_num == 1 and request.post_id:
            raise ValueError("1회차에서는 post_id를 보내면 안 됩니다.")
    
    
    async def _process_swing_1(
        self, request, db, kf1, kf2, kf3, scores, quick_feedback, total_score
    ):
        """1회차 처리: POST, ANALYSIS, FILE 생성"""
        
        post_id = str(uuid.uuid4())
        
        # POST 생성
        post = Post(
            idx=post_id,
            user_id=request.user_id,
            type="REALTIME",
            status="ANALYZING",
            total_score=total_score
        )
        db.add(post)
        db.flush()
        
        # ANALYSIS 생성
        analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=post_id,
            kf1=kf1,
            kf2=kf2,
            kf3=kf3,
            score_json=scores
        )
        db.add(analysis)
        
        # FILE 생성 (이미지 3개 + 동영상 2개)
        if request.frames and len(request.frames) > max(kf1, kf2, kf3):
            self._save_keyframe_files(
                db, post_id, request.frames, [kf1, kf2, kf3], swing_num=1
            )
        
        db.commit()
        db.refresh(post)
        
        return QuickFeedbackResponse(
            swing_num=1,
            post_id=post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            scores=ScoreDetail(**scores)
        )
    
    
    async def _process_swing_2_or_3(
        self, request, db, kf1, kf2, kf3, scores, quick_feedback, total_score
    ):
        """2~3회차 처리: POST 업데이트, ANALYSIS 업데이트, FILE 추가"""
        
        # POST 조회
        post = db.query(Post).filter(
            Post.idx == request.post_id,
            Post.user_id == request.user_id
        ).first()
        
        if not post:
            raise ValueError("POST를 찾을 수 없습니다. post_id를 확인하세요.")
        
        post.total_score = total_score
        
        # ANALYSIS 업데이트
        analysis = db.query(Analysis).filter(
            Analysis.post_idx == request.post_id
        ).first()
        
        if analysis:
            # 점수 평균 계산
            old_scores = analysis.score_json
            new_scores = {
                key: (old_scores.get(key, 0) + scores[key]) // 2
                for key in scores
            }
            analysis.score_json = new_scores
            analysis.kf1 = kf1
            analysis.kf2 = kf2
            analysis.kf3 = kf3
        
        # FILE 추가 (이미지 3개 + 동영상 2개)
        if request.frames and len(request.frames) > max(kf1, kf2, kf3):
            self._save_keyframe_files(
                db, request.post_id, request.frames,
                [kf1, kf2, kf3], swing_num=request.swing_num
            )
        
        # 3회차면 완료 처리
        if request.swing_num == 3:
            post.status = "DONE"
            post.total_score = sum(analysis.score_json.values()) // len(analysis.score_json)
            
            db.commit()
            db.refresh(post)
            
            return AnalysisCompleteResponse(
                swing_num=3,
                post_id=request.post_id,
                save_to_db=True,
                total_score=post.total_score,
                scores=ScoreDetail(**analysis.score_json),
                quick_feedback=quick_feedback
            )
        
        # 2회차
        db.commit()
        db.refresh(post)
        
        return QuickFeedbackResponse(
            swing_num=2,
            post_id=request.post_id,
            quick_feedback=quick_feedback,
            save_to_db=True,
            scores=ScoreDetail(**scores)
        )
    
    
    def _save_keyframe_files(self, db, post_id, frames, keyframe_indices, swing_num):
        """
        키프레임 저장 (이미지 3개 + 동영상 2개)
        - 이미지 3개: KF1, KF2, KF3
        - 동영상 2개: BACKSWING, IMPACT
        """
        
        kf1, kf2, kf3 = keyframe_indices
        
        print(f"\n🎬 스윙 {swing_num}회 키프레임 저장:")
        print(f"   📸 이미지 3개: KF1({kf1}), KF2({kf2}), KF3({kf3})")
        print(f"   🎥 동영상 2개: BACKSWING({kf1}→{kf2}), IMPACT({kf2}→{kf3})")
        
        # 1. 이미지 3개 저장
        for kf_num, kf_idx in enumerate([kf1, kf2, kf3], 1):
            frame_image = frames[kf_idx]
            file_path = self._save_image(post_id, kf_num, swing_num, frame_image)
            
            file = File(
                idx=str(uuid.uuid4()),
                post_idx=post_id,
                file_type=f"KF{kf_num}",
                file_name=f"swing{swing_num}_kf{kf_num}.jpg",
                file_path=file_path,
                file_extension="jpg",
                storage_type="LOCAL"
            )
            db.add(file)
            print(f"  ✅ KF{kf_num} 이미지 저장")
        
        # 2. 백스윙 동영상 (KF1 → KF2)
        backswing_path = self._save_video_from_frames(
            post_id, swing_num, frames, kf1, kf2, 'BACKSWING'
        )
        
        file = File(
            idx=str(uuid.uuid4()),
            post_idx=post_id,
            file_type="BACKSWING",
            file_name=f"swing{swing_num}_backswing.mp4",
            file_path=backswing_path,
            file_extension="mp4",
            storage_type="LOCAL"
        )
        db.add(file)
        print(f"  ✅ BACKSWING 동영상 저장 ({kf2-kf1+1}프레임)")
        
        # 3. 임팩트 동영상 (KF2 → KF3)
        impact_path = self._save_video_from_frames(
            post_id, swing_num, frames, kf2, kf3, 'IMPACT'
        )
        
        file = File(
            idx=str(uuid.uuid4()),
            post_idx=post_id,
            file_type="IMPACT",
            file_name=f"swing{swing_num}_impact.mp4",
            file_path=impact_path,
            file_extension="mp4",
            storage_type="LOCAL"
        )
        db.add(file)
        print(f"  ✅ IMPACT 동영상 저장 ({kf3-kf2+1}프레임)")
        
        print(f"✅ 스윙 {swing_num}회 저장 완료: 이미지 3개 + 동영상 2개\n")
    
    
    def _save_image(self, post_id, kf_num, swing_num, image_base64):
        """이미지 파일 저장 (backend/data/realtime)"""
        
        # 폴더 생성
        post_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(post_dir, exist_ok=True)
        
        # Base64 디코딩
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        
        # 파일 저장
        filename = f"swing{swing_num}_kf{kf_num}.jpg"
        filepath = os.path.join(post_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        # 경로 정규화 (Windows \\ → /)
        return filepath.replace("\\", "/")
    
    
    def _save_video_from_frames(self, post_id, swing_num, frames, start_idx, end_idx, video_type):
        """
        Base64 프레임들로 동영상 생성
        
        Args:
            post_id: POST UUID
            swing_num: 스윙 번호 (1, 2, 3)
            frames: Base64 이미지 리스트
            start_idx: 시작 프레임 인덱스
            end_idx: 끝 프레임 인덱스
            video_type: 'BACKSWING' 또는 'IMPACT'
        """
        
        # 폴더 생성
        post_dir = os.path.join(self.save_dir, post_id)
        os.makedirs(post_dir, exist_ok=True)
        
        # 파일명
        if video_type == 'BACKSWING':
            filename = f"swing{swing_num}_backswing.mp4"
        else:
            filename = f"swing{swing_num}_impact.mp4"
        
        filepath = os.path.join(post_dir, filename)
        
        # 첫 프레임으로 크기 확인
        first_frame_base64 = frames[start_idx].split(",")[1] if "," in frames[start_idx] else frames[start_idx]
        first_frame_data = base64.b64decode(first_frame_base64)
        first_frame_array = np.frombuffer(first_frame_data, dtype=np.uint8)
        first_frame = cv2.imdecode(first_frame_array, cv2.IMREAD_COLOR)
        
        height, width, _ = first_frame.shape
        
        # VideoWriter 설정 (30fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 30, (width, height))
        
        # 프레임 범위 동영상으로 저장
        for i in range(start_idx, end_idx + 1):
            # Base64 디코딩
            frame_base64 = frames[i].split(",")[1] if "," in frames[i] else frames[i]
            frame_data = base64.b64decode(frame_base64)
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # 프레임 쓰기
            out.write(frame)
        
        out.release()
        
        # 경로 정규화
        return filepath.replace("\\", "/")


# 싱글톤 인스턴스
swing_service = SwingService()
