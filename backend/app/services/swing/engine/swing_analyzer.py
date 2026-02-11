"""
swing_analyzer.py
=================
배드민턴 스윙 분석 핵심 엔진

구간 분할, 지표 계산, 점수화
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from math import sqrt, acos, degrees

from .pose_extractor import FrameLandmarks
from .constants import SWING_CRITERIA, PHASE_RATIOS, calculate_grade, get_overall_score


@dataclass
class PhaseInfo:
    """구간 정보"""
    name: str
    start_frame: int
    end_frame: int
    key_frame: int  # 대표 프레임 (준비=시작, 임팩트=최고점, 팔로우=끝)


@dataclass
class SwingMetrics:
    """스윙 지표"""
    elbow_angle: float          # 임팩트 순간 팔꿈치 각도
    impact_height: float        # 임팩트 높이 비율
    hip_rotation: float         # 골반 회전량
    followthrough: float        # 팔로우스루 정도


@dataclass 
class SwingAnalysisResult:
    """분석 결과"""
    phases: Dict[str, PhaseInfo]
    metrics: SwingMetrics
    scores: Dict[str, Dict]     # 지표별 점수/등급/피드백
    overall_score: int
    overall_grade: str
    impact_frame: int
    feedback_summary: List[str]


class SwingAnalyzer:
    """
    스윙 분석기
    
    사용법:
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(frames_data)
    """
    
    def __init__(self):
        pass
    
    def analyze(self, frames_data: List[FrameLandmarks]) -> SwingAnalysisResult:
        """
        전체 분석 파이프라인
        
        Args:
            frames_data: 프레임별 관절 데이터
            
        Returns:
            SwingAnalysisResult: 분석 결과
        """
        # 1. 임팩트 프레임 찾기
        impact_frame = self._find_impact_frame(frames_data)
        
        # 2. 구간 분할
        phases = self._segment_phases(frames_data, impact_frame)
        
        # 3. 지표 계산
        metrics = self._calculate_metrics(frames_data, phases)
        
        # 4. 점수화
        scores = self._calculate_scores(metrics)
        
        # 5. 종합 점수
        overall_score = get_overall_score(scores)
        overall_grade = self._score_to_grade(overall_score)
        
        # 6. 피드백 요약
        feedback_summary = self._generate_feedback_summary(scores)
        
        return SwingAnalysisResult(
            phases=phases,
            metrics=metrics,
            scores=scores,
            overall_score=overall_score,
            overall_grade=overall_grade,
            impact_frame=impact_frame,
            feedback_summary=feedback_summary
        )
    
    # ============================================================
    # 1. 임팩트 프레임 찾기
    # ============================================================
    def _find_impact_frame(self, frames_data: List[FrameLandmarks]) -> int:
        """
        임팩트 프레임 찾기
        
        방법: 손목 y좌표가 가장 낮은(=높은 위치) 프레임
        """
        if not frames_data:
            return 0
        
        wrist_y_values = []
        for frame in frames_data:
            wrist_y = frame.landmarks.get("right_wrist", (0, 0.5, 0))[1]
            wrist_y_values.append(wrist_y)
        
        # 스무딩 (노이즈 제거)
        smoothed = np.convolve(wrist_y_values, np.ones(3)/3, mode='same')
        
        # 검색 범위 제한 (30~70%)
        total = len(frames_data)
        search_start = int(total * PHASE_RATIOS["impact_search_start"])
        search_end = int(total * PHASE_RATIOS["impact_search_end"])
        
        # 해당 범위에서 최소값 찾기
        search_range = smoothed[search_start:search_end]
        if len(search_range) > 0:
            local_min_idx = int(np.argmin(search_range))
            impact_frame = search_start + local_min_idx
        else:
            impact_frame = total // 2
        
        return impact_frame
    
    # ============================================================
    # 2. 구간 분할
    # ============================================================
    def _segment_phases(self, frames_data: List[FrameLandmarks], impact_frame: int) -> Dict[str, PhaseInfo]:
        """
        3구간으로 분할
        
        - 구간1: 준비자세 (시작 ~ 백스윙 시작)
        - 구간2: 백스윙~임팩트 (백스윙 시작 ~ 임팩트+여유)
        - 구간3: 팔로우스루 (임팩트 ~ 마무리)
        """
        total = len(frames_data)
        
        # 구간 경계
        ready_end = int(total * PHASE_RATIOS["ready_end"])
        backswing_start = ready_end
        
        # 임팩트 + 약간 여유
        impact_end = min(impact_frame + 5, total - 1)
        
        # 팔로우스루
        followthrough_end = min(
            impact_frame + int(total * PHASE_RATIOS["followthrough_duration"]),
            total - 1
        )
        
        phases = {
            "ready": PhaseInfo(
                name="준비자세",
                start_frame=0,
                end_frame=ready_end,
                key_frame=ready_end // 2  # 중간 프레임
            ),
            "backswing_impact": PhaseInfo(
                name="백스윙~임팩트", 
                start_frame=backswing_start,
                end_frame=impact_end,
                key_frame=impact_frame  # 임팩트 순간
            ),
            "followthrough": PhaseInfo(
                name="팔로우스루",
                start_frame=impact_frame,
                end_frame=followthrough_end,
                key_frame=followthrough_end  # 마무리 순간
            )
        }
        
        return phases
    
    # ============================================================
    # 3. 지표 계산
    # ============================================================
    def _calculate_metrics(self, frames_data: List[FrameLandmarks], phases: Dict[str, PhaseInfo]) -> SwingMetrics:
        """모든 핵심 지표 계산"""
        
        # 핵심 프레임 추출
        ready_frame = frames_data[phases["ready"].key_frame]
        impact_frame = frames_data[phases["backswing_impact"].key_frame]
        finish_frame = frames_data[phases["followthrough"].key_frame]
        
        # 1. 팔꿈치 신전 각도 (임팩트 순간)
        elbow_angle = self._calc_elbow_angle(impact_frame.landmarks)
        
        # 2. 임팩트 높이
        impact_height = self._calc_impact_height(impact_frame.landmarks)
        
        # 3. 골반 회전량 (준비 vs 임팩트)
        hip_rotation = self._calc_hip_rotation(
            ready_frame.landmarks,
            impact_frame.landmarks
        )
        
        # 4. 팔로우스루 (임팩트 vs 마무리)
        followthrough = self._calc_followthrough(
            impact_frame.landmarks,
            finish_frame.landmarks
        )
        
        return SwingMetrics(
            elbow_angle=elbow_angle,
            impact_height=impact_height,
            hip_rotation=hip_rotation,
            followthrough=followthrough
        )
    
    def _calc_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """세 점으로 각도 계산 (p2가 꼭지점)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 * mag2 < 1e-6:
            return 0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
        return degrees(acos(cos_angle))
    
    def _calc_elbow_angle(self, landmarks: Dict) -> float:
        """
        팔꿈치 신전 각도
        
        어깨(12) - 팔꿈치(14) - 손목(16) 각도
        """
        shoulder = landmarks.get("right_shoulder", (0, 0, 0))[:2]
        elbow = landmarks.get("right_elbow", (0, 0, 0))[:2]
        wrist = landmarks.get("right_wrist", (0, 0, 0))[:2]
        
        return self._calc_angle(shoulder, elbow, wrist)
    
    def _calc_impact_height(self, landmarks: Dict) -> float:
        """
        임팩트 높이 비율
        
        (어깨y - 손목y) / 상체길이
        양수 = 손목이 어깨보다 위
        """
        shoulder = landmarks.get("right_shoulder", (0, 0.5, 0))
        wrist = landmarks.get("right_wrist", (0, 0.5, 0))
        hip = landmarks.get("right_hip", (0, 0.7, 0))
        
        body_height = abs(shoulder[1] - hip[1])
        if body_height < 0.01:
            body_height = 0.3
        
        # MediaPipe: y는 아래로 갈수록 큼
        # 어깨y - 손목y > 0 이면 손목이 위
        height_diff = shoulder[1] - wrist[1]
        
        return height_diff / body_height
    
    def _calc_hip_rotation(self, ready_lm: Dict, impact_lm: Dict) -> float:
        """
        골반 회전량
        
        (준비 골반너비 - 임팩트 골반너비) / 준비 골반너비
        양수 = 회전함
        """
        ready_hip_width = abs(
            ready_lm.get("left_hip", (0, 0, 0))[0] - 
            ready_lm.get("right_hip", (0, 0, 0))[0]
        )
        
        impact_hip_width = abs(
            impact_lm.get("left_hip", (0, 0, 0))[0] - 
            impact_lm.get("right_hip", (0, 0, 0))[0]
        )
        
        if ready_hip_width < 0.01:
            return 0
        
        return (ready_hip_width - impact_hip_width) / ready_hip_width
    
    def _calc_followthrough(self, impact_lm: Dict, finish_lm: Dict) -> float:
        """
        팔로우스루 정도
        
        (마무리 손목y - 임팩트 손목y) / 상체길이
        양수 = 손목이 내려옴 (팔로우스루 완료)
        """
        impact_wrist_y = impact_lm.get("right_wrist", (0, 0, 0))[1]
        finish_wrist_y = finish_lm.get("right_wrist", (0, 0.5, 0))[1]
        
        body_height = abs(
            impact_lm.get("right_shoulder", (0, 0.5, 0))[1] -
            impact_lm.get("right_hip", (0, 0.7, 0))[1]
        )
        
        if body_height < 0.01:
            body_height = 0.3
        
        # y가 커지면 아래로 내려간 것
        descent = finish_wrist_y - impact_wrist_y
        
        return descent / body_height
    
    # ============================================================
    # 4. 점수화
    # ============================================================
    def _calculate_scores(self, metrics: SwingMetrics) -> Dict[str, Dict]:
        """각 지표별 점수 계산"""
        return {
            "elbow_angle": calculate_grade(metrics.elbow_angle, "elbow_angle"),
            "impact_height": calculate_grade(metrics.impact_height, "impact_height"),
            "hip_rotation": calculate_grade(metrics.hip_rotation, "hip_rotation"),
            "followthrough": calculate_grade(metrics.followthrough, "followthrough"),
        }
    
    def _score_to_grade(self, score: int) -> str:
        """점수 → 등급"""
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    # ============================================================
    # 5. 피드백 생성
    # ============================================================
    def _generate_feedback_summary(self, scores: Dict[str, Dict]) -> List[str]:
        """개선이 필요한 항목 피드백"""
        feedback = []
        
        # 점수가 낮은 순서대로 정렬
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1]["score"]
        )
        
        # 상위 2개 문제점 피드백
        for key, score_data in sorted_scores[:2]:
            if score_data["grade"] != "good":
                criteria_name = SWING_CRITERIA[key]["name"]
                feedback.append(f"[{criteria_name}] {score_data['feedback']}")
        
        if not feedback:
            feedback.append("전반적으로 좋은 스윙이에요! 계속 유지하세요 👍")
        
        return feedback


# ============================================================
# GT 데이터 생성기
# ============================================================
class GTExtractor:
    """
    전문가 영상에서 GT 데이터 추출
    """
    
    def __init__(self):
        self.analyzer = SwingAnalyzer()
    
    def extract_from_video(self, video_path: str, frames_data: List[FrameLandmarks]) -> Dict:
        """
        전문가 영상 1개에서 GT 지표 추출
        
        Returns:
            dict: 지표값들
        """
        result = self.analyzer.analyze(frames_data)
        
        return {
            "video_path": video_path,
            "total_frames": len(frames_data),
            "impact_frame": result.impact_frame,
            "elbow_angle": result.metrics.elbow_angle,
            "impact_height": result.metrics.impact_height,
            "hip_rotation": result.metrics.hip_rotation,
            "followthrough": result.metrics.followthrough,
        }
    
    def calculate_criteria_from_gt_list(self, gt_list: List[Dict]) -> Dict:
        """
        GT 리스트에서 판정 기준 도출
        
        Args:
            gt_list: 전문가별 GT 데이터 리스트
            
        Returns:
            dict: 판정 기준 (min, max, good, fair 범위)
        """
        metrics_names = ["elbow_angle", "impact_height", "hip_rotation", "followthrough"]
        
        criteria = {}
        
        for metric in metrics_names:
            values = [gt[metric] for gt in gt_list if metric in gt]
            
            if not values:
                continue
            
            min_val = min(values)
            max_val = max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # good 범위: 전문가 범위 + 10% 여유
            margin = (max_val - min_val) * 0.15
            good_min = min_val - margin
            good_max = max_val + margin
            
            # fair 범위: good보다 20% 더 넓게
            fair_margin = (max_val - min_val) * 0.3
            fair_min = min_val - fair_margin
            fair_max = max_val + fair_margin
            
            criteria[metric] = {
                "expert_min": round(min_val, 3),
                "expert_max": round(max_val, 3),
                "expert_mean": round(mean_val, 3),
                "expert_std": round(std_val, 3),
                "good": (round(good_min, 3), round(good_max, 3)),
                "fair": (round(fair_min, 3), round(fair_max, 3)),
            }
        
        return criteria
