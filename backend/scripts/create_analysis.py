# backend/scripts/create_analysis.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.userModels import User
from app.models.postModels import Post
from app.models.analysisModels import Analysis

db = SessionLocal()

def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))

def score_to_err(score: float, lo: float = 0.02, hi: float = 0.20) -> float:
    s = clamp(float(score), 0.0, 100.0)
    base = hi - (s / 100.0) * (hi - lo)
    noise = random.uniform(-0.005, 0.005)
    return round(clamp(base + noise, lo, hi), 4)

def make_metric(measured: float | None, target: str | None, diff: float | None, score: float) -> dict:
    d = {"score": float(clamp(score, 0.0, 100.0))}
    if measured is not None: d["measured"] = measured
    if target is not None: d["target"] = target
    if diff is not None: d["diff"] = diff
    return d

def mean_score(scores: list[float]) -> float:
    if not scores: return 0.0
    return round(sum(scores) / len(scores), 2)

def build_post_kwargs(post_idx: str, created_at: datetime, user_id: str) -> dict:
    cols = list(Post.__table__.columns)
    kwargs: dict = {"idx": post_idx}
    
    # user_id 필드 대응
    if any(c.name == "user_id" for c in cols): kwargs["user_id"] = user_id
    
    # 날짜 필드 대응 (create_date 또는 created_at)
    if any(c.name == "create_date" for c in cols): kwargs["create_date"] = created_at
    if any(c.name == "created_at" for c in cols): kwargs["created_at"] = created_at

    # 기타 필수 필드 더미 채우기
    for c in cols:
        name = c.name
        if name in kwargs or c.nullable or c.default is not None or c.server_default is not None:
            continue
        if name in ("sport", "action"): kwargs[name] = "badminton"
        elif name in ("title", "name"): kwargs[name] = "dummy analysis"
        else: kwargs[name] = ""
    return kwargs

try:
    # 🌟 여기에 팀장님의 실제 DB에 있는 사용자 ID를 적어주세요.
    USER_ID = "admin" 
    TOTAL_DAYS = 300

    print(f"🚀 {USER_ID} 사용자를 위한 {TOTAL_DAYS}일치 데이터 생성을 시작합니다...")

    for days_ago in range(TOTAL_DAYS):
        # 날짜 계산
        dt = datetime.now() - timedelta(days=days_ago)
        post_idx_value = str(uuid.uuid4())

        # 성장 곡선 (과거 80점 -> 현재 95점)
        progress = (TOTAL_DAYS - 1 - days_ago) / float(TOTAL_DAYS - 1)
        base_avg = 80.0 + (15.0 * progress)

        # 1. Post(부모) 데이터 생성 및 추가
        post_data = build_post_kwargs(post_idx_value, dt, USER_ID)
        new_post = Post(**post_data)
        db.add(new_post)
        
        # ⭐ [핵심 수정] Analysis를 넣기 전에 Post를 먼저 DB 세션에 등록(Flush)하여 에러 방지
        db.flush() 

        # 2. Analysis(자식) 상세 점수 생성
        ready_score = clamp(random.gauss(base_avg, 3.0))
        rotation_score = clamp(random.gauss(base_avg, 3.0))
        backswing_score = clamp(random.gauss(base_avg - 5, 5.0)) # 백스윙이 좀 약한 설정
        impact_score = clamp(random.gauss(base_avg, 3.0))
        
        score_json = {
            "total_score": mean_score([ready_score, rotation_score, backswing_score, impact_score]),
            "details": {
                "Ready": {"Ready_score": ready_score},
                "Rotation": {"Rotation_score": rotation_score},
                "Backswing": {"Backswing_score": backswing_score},
                "Impact": {"Impact_score": impact_score},
                "FollowSwing": {"FollowSwing_score": 100.0, "Performance": {"success": True}}
            }
        }

        # 3. Analysis 객체 생성 및 추가
        new_analysis = Analysis(
            idx=str(uuid.uuid4()),
            post_idx=post_idx_value,
            kf1=1, kf2=2, kf3=3,
            kf1_error=score_to_err(ready_score),
            kf2_error=score_to_err(backswing_score),
            kf3_error=score_to_err(impact_score),
            score_json=score_json,
            create_date=dt
        )
        db.add(new_analysis)

    # 4. 전체 데이터 일괄 확정
    db.commit()
    print(f"✅ 생성 완료! user_id={USER_ID} 기준 {TOTAL_DAYS}개의 게시물과 분석 결과가 저장되었습니다.")

except Exception as e:
    db.rollback()
    print(f"❌ 실패: {e}")
finally:
    db.close()