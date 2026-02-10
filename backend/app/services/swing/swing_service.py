"""
스윙 분석 서비스 (swing.py용)
"""

class SwingService:
    
    def detect_keyframes(self, keypoints):
        """
        키프레임 감지 (임시 더미)
        
        Args:
            keypoints: List[List[float]] - 33개 랜드마크 x 4개 값
        
        Returns:
            tuple: (kf1, kf2, kf3) - 프레임 인덱스
        """
        # TODO: 실제 알고리즘 구현
        total_frames = len(keypoints)
        
        kf1 = total_frames // 3      # 준비자세
        kf2 = total_frames * 2 // 3  # 백스윙
        kf3 = total_frames - 1       # 임팩트
        
        return kf1, kf2, kf3
    
    
    def calculate_scores(self, keypoints, kf1, kf2, kf3):
        """
        점수 계산 (임시 더미)
        
        Args:
            keypoints: List[List[float]]
            kf1, kf2, kf3: 키프레임 인덱스
        
        Returns:
            dict: 6대 지표 점수
        """
        # TODO: 실제 점수 계산 알고리즘 구현
        return {
            "elbow_height": 85,
            "wrist_snap": 78,
            "hit_position": 90,
            "shoulder_rotation": 82,
            "racket_angle": 88,
            "follow_through": 75
        }
    
    
    def get_quick_feedback(self, scores):
        """
        빠른 피드백 생성
        
        Args:
            scores: dict - 6대 지표 점수
        
        Returns:
            str: 피드백 메시지
        """
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
            overall = "아주 좋아요! 조금만 더 보완하면 완벽해질 거예요! 👍"
        elif avg >= 70:
            overall = "괜찮아요! 몇 가지 개선 포인트가 있네요. 💪"
        else:
            overall = "기본기부터 다시 연습해봐요! 천천히 개선해나가요. 📈"
        
        # ⭐ summary 제거!
        return {
            "overall": overall,
            "details": scores,
            "strengths": ["팔꿈치 높이가 좋아요!", "타구 위치가 정확해요!"],
            "improvements": ["손목 스냅을 조금 더 활용해보세요.", "팔로우스루를 더 크게!"],
            "next_goals": ["손목 스냅 80점 → 85점", "전체 평균 90점 달성"]
        }


# 싱글톤 인스턴스
swing_service = SwingService()