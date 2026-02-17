# # backend/scripts/create_analysis.py
# import sys
# from pathlib import Path
# from datetime import datetime
# import uuid

# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))

# from app.db.session import SessionLocal
# from app.models.postModels import Post  # 테이블 등록용 (필수)
# # ✅ 여기만 "실제 Analysis 모델 경로"에 맞게 바꾸세요
# # 예: from app.models.analysisModels import Analysis
# from app.models.analysisModels import Analysis  # <-- 이 줄은 프로젝트에 맞게 수정 필요


# db = SessionLocal()

# try:
#     # 1) 예시용 post_idx (실제 존재하는 POST.idx 값으로 수정하세요)
#     post_idx_value = "d6072ac0-2de6-47ad-9067-03bfe68f9d32"

#     # 2) analysis 레코드 생성 (테이블 스키마 기준)
#     analysis = Analysis(
#         idx=str(uuid.uuid4()),         # PK UUID 생성
#         post_idx=post_idx_value,       # FK (POST.idx)
#         kf1=1,
#         kf2=2,
#         kf3=3,
#         kf1_error=0.12,
#         kf2_error=0.08,
#         kf3_error=0.15,
#         score_json={
#             "ready": 85,
#             "backswing": 78,
#             "impact": 90,
#             "rotation": 88,
#             "balance": 80,
#             "followthrough": 92
#         },
#         create_date=datetime.utcnow()
#     )

#     # 3) DB 저장
#     db.add(analysis)
#     db.commit()
#     print("✅ analysis 레코드 생성 완료!")

# except Exception as e:
#     db.rollback()
#     print(f"❌ 실패: {e}")

# finally:
#     db.close()

# backend/scripts/create_analysis.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.postModels import Post  # 테이블 등록용
from app.models.analysisModels import Analysis

db = SessionLocal()

try:
    post_idx_value = "e70ba785-e705-41f8-b83c-c9ea45da28f5"

    # ✅ 40일치 생성 (1개월(30d) + 여유분) : 0~39일 전
    #    - 1주일(7d) vs 1개월(30d) 비교가 되도록 최소 한 달 이상 데이터 확보
    rows = []
    for days_ago in range(40):
        dt = datetime.utcnow() - timedelta(days=days_ago)
        rows.append(
            Analysis(
                idx=str(uuid.uuid4()),
                post_idx=post_idx_value,
                kf1=1, kf2=2, kf3=3,
                # 최근(0일 전)일수록 오차가 작고, 과거로 갈수록 오차가 커지는 형태(개선 추세)
                kf1_error=round(0.10 + (days_ago * 0.002), 4),
                kf2_error=round(0.08 + (days_ago * 0.0022), 4),
                kf3_error=round(0.12 + (days_ago * 0.0018), 4),
                score_json={
                    "ready": max(0, min(100, 75 + (39 - days_ago))),
                    "backswing": max(0, min(100, 70 + (39 - days_ago))),
                    "impact": max(0, min(100, 80 + (39 - days_ago))),
                    "rotation": max(0, min(100, 78 + (39 - days_ago))),
                    "balance": max(0, min(100, 72 + (39 - days_ago))),
                    "followthrough": max(0, min(100, 82 + (39 - days_ago))),
                },
                create_date=dt,  # ✅ 날짜별 create_date 강제 입력
            )
        )

    db.add_all(rows)
    db.commit()
    print(f"✅ analysis 40일치({len(rows)}건) 생성 완료! post_idx={post_idx_value}")

except Exception as e:
    db.rollback()
    print(f"❌ 실패: {e}")

finally:
    db.close()