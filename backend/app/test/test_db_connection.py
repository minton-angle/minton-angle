
"""PostgreSQL 연결 테스트"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# .env 파일 로드 (backend/.env)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

print("=" * 60)
print("PostgreSQL 연결 테스트")
print("=" * 60)
print(f"\n📍 DATABASE_URL: {DATABASE_URL}\n")

try:
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        
        print("✅ PostgreSQL 연결 성공!")
        print("=" * 60)
        print(f"\n버전:\n{version}\n")
        
        result = conn.execute(text("SELECT current_database();"))
        db_name = result.fetchone()[0]
        print(f"현재 데이터베이스: {db_name}")
        
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """))
        tables = result.fetchall()
        
        print(f"\n테이블 개수: {len(tables)}개")
        if tables:
            print("\n테이블 목록:")
            for table in tables:
                print(f"  - {table[0]}")
        else:
            print("\n테이블: 없음 (정상 - 아직 생성 전)")
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 통과!")
        print("=" * 60)

except Exception as e:
    print("\n❌ PostgreSQL 연결 실패!")
    print("=" * 60)
    print(f"\n에러: {e}\n")
    print("\n" + "=" * 60)