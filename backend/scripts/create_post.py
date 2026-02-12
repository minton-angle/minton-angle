# backend/scripts/create_post.py
import sys
from pathlib import Path
from datetime import datetime
import uuid

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.userModels import User

# ✅ 프로젝트에 맞게 수정하세요 (Post 모델 실제 위치)
# 예: from app.models.postModels import Post
from app.models.postModels import Post  # <-- 여기만 맞추면 됩니다

db = SessionLocal()

try:
    # ==========================
    # 1) FK로 쓸 유저 (반드시 존재해야 함)
    # ==========================
    user_id_value = "user_001"

    user = db.query(User).filter(User.id == user_id_value).first()
    if not user:
        raise RuntimeError(f"❌ User not found: {user_id_value} (POST 생성 불가)")

    # ==========================
    # 2) POST 생성 (스키마 기반)
    # ==========================
    post = Post(
        idx=str(uuid.uuid4()),        # PK UUID
        user_id=user_id_value,        # FK (USER.id)
        type="VIDEO",                # REALTIME / VIDEO (예시)
        status="UPLOADED",           # UPLOADED / ANALYZING / DONE (예시)
        total_score=None,            # 예시 (원호님 로직에 맞게)
        create_date=datetime.utcnow()
    )

    db.add(post)
    db.commit()

    print("✅ POST 생성 완료!")
    print(f"   - post.idx: {post.idx}")
    print(f"   - post.user_id: {post.user_id}")
    print(f"   - post.type: {post.type}")
    print(f"   - post.status: {post.status}")
    print(f"   - post.total_score: {post.total_score}")
    print(f"   - post.create_date: {post.create_date}")

except Exception as e:
    db.rollback()
    print(f"❌ 실패: {e}")

finally:
    db.close()