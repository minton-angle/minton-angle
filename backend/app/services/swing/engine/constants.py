"""
constants.py
============
스윙 분석 핵심 상수 및 판정 기준

전문가 GT 10개 영상 분석 기반으로 도출된 기준값
"""

# ============================================================
# MediaPipe 관절 인덱스 (33개 중 핵심만)
# ============================================================
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


# ============================================================
# 구간별 핵심 지표 정의
# ============================================================
"""
┌─────────────────────────────────────────────────────────────────────────┐
│                        기본 스윙 3단계 분석                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [구간1] 준비자세 (Ready Position) - 고정 프레임 1장                     │
│  ├─ 측정: 스탠스, 무릎 굽힘, 라켓 위치                                   │
│  └─ 특징: 스윙 시작 전 기본 자세                                         │
│                                                                         │
│  [구간2] 백스윙 ~ 임팩트 (Backswing to Impact) - 영상 클립               │
│  ├─ 측정: 팔꿈치 신전, 임팩트 높이, 골반 회전                            │
│  └─ 특징: 힘 생성 & 타격                                                │
│                                                                         │
│  [구간3] 팔로우스루 (Follow-through) - 영상 클립                         │
│  ├─ 측정: 팔로우스루 완료 여부                                           │
│  └─ 특징: 타격 후 마무리                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================
# 판정 기준 (전문가 GT 기반)
# ============================================================

SWING_CRITERIA = {
    # ========================================
    # 구간2: 임팩트 순간 팔꿈치 신전 각도
    # ========================================
    "elbow_angle": {
        "name": "팔꿈치 신전",
        "description": "임팩트 순간 어깨-팔꿈치-손목 각도",
        "unit": "degrees",
        "good": (155, 175),      # 전문가 범위 + 여유
        "fair": (140, 185),
        "expert_range": (160, 175),  # 실제 전문가 범위
        "feedback": {
            "low": "팔을 더 쭉 펴주세요! 힘이 제대로 전달되지 않아요.",
            "high": "팔이 과하게 펴졌어요. 자연스럽게 펴주세요.",
            "good": "팔꿈치 각도 완벽해요! 👍"
        }
    },
    
    # ========================================
    # 구간2: 임팩트 높이 (손목 위치)
    # ========================================
    "impact_height": {
        "name": "임팩트 높이",
        "description": "손목이 어깨 대비 얼마나 위에 있는지",
        "unit": "ratio",
        "good": (0.3, 0.7),       # 어깨보다 30~70% 위
        "fair": (0.15, 0.85),
        "expert_range": (0.4, 0.65),
        "feedback": {
            "low": "더 높은 곳에서 치세요! 타점이 낮으면 공이 안 떠요.",
            "high": "타점이 너무 높아요. 자연스러운 높이에서 치세요.",
            "good": "타점 높이 좋아요! 👍"
        }
    },
    
    # ========================================
    # 구간2: 골반 회전량
    # ========================================
    "hip_rotation": {
        "name": "골반 회전",
        "description": "준비자세 대비 임팩트 시 골반 회전 정도",
        "unit": "ratio",
        "good": (0.15, 0.40),     # 15~40% 회전
        "fair": (0.08, 0.50),
        "expert_range": (0.18, 0.35),
        "feedback": {
            "low": "골반을 더 회전시켜주세요! 하체 힘이 안 실려요.",
            "high": "골반 회전이 과해요. 균형을 잡아주세요.",
            "good": "골반 회전 좋아요! 하체 힘이 잘 전달되고 있어요 👍"
        }
    },
    
    # ========================================
    # 구간3: 팔로우스루 완료
    # ========================================
    "followthrough": {
        "name": "팔로우스루",
        "description": "임팩트 후 손목이 충분히 내려왔는지",
        "unit": "ratio",
        "good": (0.25, 0.60),     # 상체 길이의 25~60%
        "fair": (0.15, 0.75),
        "expert_range": (0.30, 0.50),
        "feedback": {
            "low": "끝까지 휘둘러주세요! 중간에 멈추면 부상 위험이 있어요.",
            "high": "팔로우스루가 과해요. 자연스럽게 마무리하세요.",
            "good": "팔로우스루 완벽해요! 👍"
        }
    },
}


# ============================================================
# 구간 분할 비율
# ============================================================
PHASE_RATIOS = {
    "ready_end": 0.15,           # 0~15% = 준비자세
    "backswing_start": 0.15,     # 15% = 백스윙 시작
    "impact_search_start": 0.30, # 30~70% 구간에서 임팩트 탐색
    "impact_search_end": 0.70,
    "followthrough_duration": 0.25,  # 임팩트 후 25% = 팔로우스루
}


# ============================================================
# 점수 계산
# ============================================================
def calculate_grade(value, criteria_key):
    """
    값을 기준으로 등급 계산
    
    Returns:
        dict: {grade, score, feedback}
    """
    criteria = SWING_CRITERIA.get(criteria_key)
    if not criteria:
        return {"grade": "unknown", "score": 0, "feedback": ""}
    
    good_min, good_max = criteria["good"]
    fair_min, fair_max = criteria["fair"]
    
    if good_min <= value <= good_max:
        # Good 범위 내에서 중심에 가까울수록 높은 점수
        center = (good_min + good_max) / 2
        distance = abs(value - center) / ((good_max - good_min) / 2)
        score = int(100 - distance * 15)  # 85~100점
        return {
            "grade": "good",
            "score": score,
            "feedback": criteria["feedback"]["good"]
        }
    
    elif fair_min <= value <= fair_max:
        # Fair 범위
        if value < good_min:
            distance = (good_min - value) / (good_min - fair_min)
        else:
            distance = (value - good_max) / (fair_max - good_max)
        score = int(70 - distance * 20)  # 50~70점
        
        feedback_key = "low" if value < good_min else "high"
        return {
            "grade": "fair",
            "score": score,
            "feedback": criteria["feedback"][feedback_key]
        }
    
    else:
        # Poor 범위
        feedback_key = "low" if value < fair_min else "high"
        return {
            "grade": "poor",
            "score": 30,
            "feedback": criteria["feedback"][feedback_key]
        }


def get_overall_score(metrics_scores):
    """
    전체 점수 계산 (가중 평균)
    
    가중치:
    - 팔꿈치 신전: 35% (가장 중요)
    - 임팩트 높이: 25%
    - 골반 회전: 20%
    - 팔로우스루: 20%
    """
    weights = {
        "elbow_angle": 0.35,
        "impact_height": 0.25,
        "hip_rotation": 0.20,
        "followthrough": 0.20,
    }
    
    total_score = 0
    total_weight = 0
    
    for key, weight in weights.items():
        if key in metrics_scores:
            total_score += metrics_scores[key]["score"] * weight
            total_weight += weight
    
    if total_weight > 0:
        return int(total_score / total_weight * total_weight)
    return 0


# ============================================================
# 프론트엔드 동기화용 내보내기
# ============================================================
def export_criteria_for_frontend():
    """
    프론트엔드 JavaScript와 동기화할 기준값 반환
    """
    return {
        "SWING_CRITERIA": {
            key: {
                "good": criteria["good"],
                "fair": criteria["fair"],
                "feedback": criteria["feedback"]
            }
            for key, criteria in SWING_CRITERIA.items()
        },
        "LANDMARKS": LANDMARKS
    }
