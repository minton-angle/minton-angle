# backend/scripts/create_analysis.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.postModels import Post  # 테이블 등록용
from app.models.analysisModels import Analysis

db = SessionLocal()

try:
    post_idx_value = "e70ba785-e705-41f8-b83c-c9ea45da28f5"

    # ✅ 60일치 생성 (1개월(30d) + 여유분) : 0~39일 전
    #    - 1주일(7d) vs 1개월(30d) 비교가 되도록 최소 한 달 이상 데이터 확보
    rows = []
    for days_ago in range(300):
        dt = datetime.utcnow() - timedelta(days=days_ago)
        # ------------------------------
        # 1) 현실적인 분포 + 40일 구간에서 약 10점 개선(최신이 높음)
        #    - 39일 전: base 약 85점
        #    - 0일 전 : base 약 95점
        # ------------------------------
        def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
            return max(lo, min(hi, v))

        def to_10(v: float) -> int:
            """Quantize to 10-point increments within [0,100]."""
            return int(clamp(round(v / 10.0) * 10.0, 0.0, 100.0))

        progress = (39 - days_ago) / 39.0  # 0.0(과거) -> 1.0(최신)
        base_avg = 85.0 + (10.0 * progress)  # 약 10점 차이

        # Total 점수들(0~100): base를 중심으로 약간의 변동
        ready_total_seed = clamp(random.gauss(base_avg + 1.0, 3.0))
        rotation_total_seed = clamp(random.gauss(base_avg - 0.5, 3.0))
        backswing_total_seed = clamp(random.gauss(base_avg - 2.0, 4.0))  # backswing은 조금 흔들리게
        impact_total_seed = clamp(random.gauss(base_avg + 0.5, 3.0))
        follow_total_seed = clamp(random.gauss(base_avg + 0.0, 3.0))

        # ------------------------------
        # 2) 세부 항목 생성 (Total 규칙을 보장)
        #    - Ready_Total = (Arm, Height, Stance) 평균
        #    - Backswing_Total = (WristX, Racket, Elbow) 평균
        #    - FollowSwing_Total = (Move, Cross) 평균
        # ------------------------------
        # Ready (3개 평균)
        ready_arm = to_10(random.gauss(ready_total_seed, 6.0))
        ready_height = to_10(random.gauss(ready_total_seed, 6.0))
        ready_stance = to_10(random.gauss(ready_total_seed, 6.0))
        ready_total = round((ready_arm + ready_height + ready_stance) / 3.0, 2)

        # Rotation (단일 항목)
        rotation_hip = to_10(random.gauss(rotation_total_seed, 5.0))
        rotation_total = round(float(rotation_hip), 2)

        # Backswing (3개 평균)
        backswing_wristx = to_10(random.gauss(backswing_total_seed, 10.0))
        backswing_racket = to_10(random.gauss(backswing_total_seed, 10.0))
        backswing_elbow = to_10(random.gauss(backswing_total_seed, 10.0))
        backswing_total = round((backswing_wristx + backswing_racket + backswing_elbow) / 3.0, 2)

        # Impact (단일 항목)
        impact_angle = to_10(random.gauss(impact_total_seed, 5.0))
        impact_total = round(float(impact_angle), 2)

        # FollowSwing (boolean)
        # - 실제 서비스에서는 True/False로 들어올 예정
        # - 더미데이터는 랜덤으로 생성
        follow_pass = bool(random.choice([True, False]))

        # 하위호환: 기존 UI/API는 0~100 점수형 Total을 기대하므로
        # True면 100, False면 0으로 유지
        follow_total = 100.0 if follow_pass else 0.0

        # ------------------------------
        # 3) Average_Score
        #    - 점수 기반 Total들만 평균 (FollowSwing은 boolean이므로 제외)
        # ------------------------------
        total_items = [ready_total, rotation_total, backswing_total, impact_total]
        average_score = round(sum(total_items) / len(total_items), 2)

        # ------------------------------
        # 4) kf 오차는 점수와 반비례하도록(대충) 생성
        #    - 점수가 높을수록 오차가 낮아지게
        # ------------------------------
        def score_to_err(score: float, lo: float = 0.02, hi: float = 0.20) -> float:
            s = max(0.0, min(100.0, float(score)))
            base = hi - (s / 100.0) * (hi - lo)
            noise = random.uniform(-0.005, 0.005)
            return round(max(lo, min(hi, base + noise)), 4)

        kf1_err = score_to_err(ready_total)
        kf2_err = score_to_err(backswing_total)
        kf3_err = score_to_err(impact_total)

        score_json = {
            "1_Ready_Total": ready_total,
            "1_Ready_Arm": ready_arm,
            "1_Ready_Height": ready_height,
            "1_Ready_Stance": ready_stance,
            "2_Rotation_Total": rotation_total,
            "2_Rotation_Hip": rotation_hip,
            "3_Backswing_Total": backswing_total,
            "3_Backswing_WristX": backswing_wristx,
            "3_Backswing_Racket": backswing_racket,
            "3_Backswing_Elbow": backswing_elbow,
            "4_Impact_Total": impact_total,
            "4_Impact_Angle": impact_angle,
            "5_FollowSwing_Total": follow_total,      # 하위호환(0/100)
            "5_FollowSwing_Pass": bool(follow_pass),  # 신규(boolean)
            "Average_Score": average_score,
        }

        rows.append(
            Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_idx_value,
                kf1=1, kf2=2, kf3=3,
                kf1_error=kf1_err,
                kf2_error=kf2_err,
                kf3_error=kf3_err,
                score_json=score_json,
                create_date=dt,
            )
        )

    db.add_all(rows)
    db.commit()
    print(f"✅ analysis 300일치({len(rows)}건) 생성 완료! post_idx={post_idx_value}")

except Exception as e:
    db.rollback()
    print(f"❌ 실패: {e}")

finally:
    db.close()