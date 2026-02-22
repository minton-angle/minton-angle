# backend/scripts/create_analysis.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.postModels import Post  # 테이블 등록용(임포트 유지)
from app.models.analysisModels import Analysis

db = SessionLocal()


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def score_to_err(score: float, lo: float = 0.02, hi: float = 0.20) -> float:
    """점수가 높을수록 kf_error는 낮아지는 형태(legacy)"""
    s = clamp(float(score), 0.0, 100.0)
    base = hi - (s / 100.0) * (hi - lo)
    noise = random.uniform(-0.005, 0.005)
    return round(clamp(base + noise, lo, hi), 4)


def make_metric(measured: float | None, target: str | None, diff: float | None, score: float) -> dict:
    d = {"score": float(clamp(score, 0.0, 100.0))}
    if measured is not None:
        d["measured"] = measured
    if target is not None:
        d["target"] = target
    if diff is not None:
        d["diff"] = diff
    return d


def mean_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


try:
    post_idx_value = "e70ba785-e705-41f8-b83c-c9ea45da28f5"
    TOTAL_DAYS = 300
    rows = []

    for days_ago in range(TOTAL_DAYS):
        dt = datetime.utcnow() - timedelta(days=days_ago)

        # 0(과거) -> 1(최신)
        progress = (TOTAL_DAYS - 1 - days_ago) / float(TOTAL_DAYS - 1)
        base_avg = 85.0 + (10.0 * progress)  # 과거 85 -> 최신 95

        # stage seed
        ready_seed = clamp(random.gauss(base_avg + 1.0, 3.0))
        rotation_seed = clamp(random.gauss(base_avg - 0.5, 3.0))
        backswing_seed = clamp(random.gauss(base_avg - 2.0, 4.0))
        impact_seed = clamp(random.gauss(base_avg + 0.5, 3.0))

        # ------------------------------
        # Ready (4 metrics)
        # ------------------------------
        # 점수: seed를 중심으로 흔들리게
        ready_arm_score = clamp(random.gauss(ready_seed, 6.0))
        ready_left_wrist_score = clamp(random.gauss(ready_seed, 6.0))
        ready_stance_score = clamp(random.gauss(ready_seed, 6.0))
        # Wrist_Height_Ratio는 일부러 변동 크게(가끔 낮아지게)
        ready_ratio_score = clamp(random.gauss(ready_seed - 10.0, 12.0))

        ready_score = mean_score([ready_arm_score, ready_left_wrist_score, ready_stance_score, ready_ratio_score])

        ready_node = {
            "Ready_score": ready_score,
            "Arm_Angle": make_metric(
                measured=round(random.uniform(10.0, 80.0), 2),
                target="18 ~ 70",
                diff=0.0,
                score=ready_arm_score,
            ),
            "Left_Wrist_Height": make_metric(
                measured=round(random.uniform(-0.25, 0.15), 4),
                target="< 0",
                diff=0.0,
                score=ready_left_wrist_score,
            ),
            "Stance_Width": make_metric(
                measured=round(random.uniform(0.10, 0.32), 4),
                target="> 0.158",
                diff=0.0,
                score=ready_stance_score,
            ),
            "Wrist_Height_Ratio": make_metric(
                measured=round(random.uniform(-0.60, 0.10), 2),
                target="-0.5 ~ -0.3",
                # 예시처럼 diff가 있을 수 있으니 적당히 생성
                diff=round(random.uniform(0.0, 0.6), 2),
                score=ready_ratio_score,
            ),
        }

        # ------------------------------
        # Rotation (2 metrics)
        # ------------------------------
        rot_hip_score = clamp(random.gauss(rotation_seed, 6.0))
        rot_shoulder_score = clamp(random.gauss(rotation_seed, 6.0))
        rotation_score = mean_score([rot_hip_score, rot_shoulder_score])

        rotation_node = {
            "Rotation_score": rotation_score,
            "Hip_Level": make_metric(measured=None, target=None, diff=None, score=rot_hip_score),
            "Shoulder_Ratio": make_metric(measured=None, target=None, diff=None, score=rot_shoulder_score),
        }

        # ------------------------------
        # Backswing (3 metrics)
        # ------------------------------
        bs_wrist_score = clamp(random.gauss(backswing_seed, 10.0))
        bs_elbow_score = clamp(random.gauss(backswing_seed - 8.0, 14.0))  # elbow lift는 좀 더 흔들리게
        bs_lshape_score = clamp(random.gauss(backswing_seed, 10.0))
        backswing_score = mean_score([bs_wrist_score, bs_elbow_score, bs_lshape_score])

        backswing_node = {
            "Backswing_score": backswing_score,
            "Wrist_X_Depth": make_metric(measured=None, target=None, diff=None, score=bs_wrist_score),
            "Elbow_Lift": make_metric(measured=None, target=None, diff=None, score=bs_elbow_score),
            "L_Shape_Angle": make_metric(measured=None, target=None, diff=None, score=bs_lshape_score),
        }

        # ------------------------------
        # Impact (2 metrics)
        # ------------------------------
        imp_ext_score = clamp(random.gauss(impact_seed, 8.0))
        # Impact의 Wrist_Height_Ratio는 일부러 낮게 나올 때 있게
        imp_ratio_score = clamp(random.gauss(impact_seed - 15.0, 15.0))
        impact_score = mean_score([imp_ext_score, imp_ratio_score])

        impact_node = {
            "Impact_score": impact_score,
            "Arm_Extension_Angle": make_metric(
                measured=round(random.uniform(120.0, 185.0), 2),
                target="140 ~ 180",
                diff=0.0,
                score=imp_ext_score,
            ),
            "Wrist_Height_Ratio": make_metric(
                measured=round(random.uniform(-0.20, 0.20), 2),
                target="2.5 ~ 4.5",
                diff=round(random.uniform(0.0, 3.5), 2),
                score=imp_ratio_score,
            ),
        }

        # ------------------------------
        # FollowSwing (Performance: score + success)
        # ------------------------------
        p_success = clamp(0.45 + 0.4 * progress, 0.05, 0.95)
        follow_success = random.random() < p_success
        follow_score = 100.0 if follow_success else 0.0

        followswing_node = {
            "FollowSwing_score": follow_score,
            "Performance": {
                "score": follow_score,
                "success": bool(follow_success),
            },
        }

        # ------------------------------
        # total_score: Ready/Rotation/Backswing/Impact 평균
        # ------------------------------
        total_score = mean_score([ready_score, rotation_score, backswing_score, impact_score])

        score_json = {
            "total_score": total_score,
            "details": {
                "Ready": ready_node,
                "Rotation": rotation_node,
                "Backswing": backswing_node,
                "Impact": impact_node,
                "FollowSwing": followswing_node,
            },
        }

        # legacy kf_error는 대충 대응(서비스 과거 경로 깨지지 않게)
        kf1_err = score_to_err(ready_score)
        kf2_err = score_to_err(backswing_score)
        kf3_err = score_to_err(impact_score)

        rows.append(
            Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_idx_value,
                kf1=1,
                kf2=2,
                kf3=3,
                kf1_error=kf1_err,
                kf2_error=kf2_err,
                kf3_error=kf3_err,
                score_json=score_json,
                create_date=dt,
            )
        )

    db.add_all(rows)
    db.commit()
    print(f"✅ analysis {TOTAL_DAYS}일치({len(rows)}건) 생성 완료! post_idx={post_idx_value}")

except Exception as e:
    db.rollback()
    print(f"❌ 실패: {e}")

finally:
    db.close()