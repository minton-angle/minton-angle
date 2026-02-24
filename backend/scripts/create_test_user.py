# backend/scripts/create_test_user.py
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.db.session import SessionLocal
from app.models.userModels import User
from datetime import date

db = SessionLocal()

# user_001 생성
user = User(
    id="user_001",
    name="테스트유저",
    password="test123",
    sex="female",
    hand="right",
    create_date=date.today()
)

db.add(user)
db.commit()
print("✅ user_001 생성 완료!")
db.close()