"""
MINTON-ANGLE API Server
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# ⭐ 먼저 서버 시작 메시지 출력 (import 전!)
import os
import logging
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ⭐ 서버 시작 메시지 (import 전에 출력)
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 MINTON-ANGLE API Server 시작")
    print("=" * 60)
    print(f"📁 작업 디렉토리: {current_dir}")
    print(f"🌐 서버 주소: http://localhost:8000")
    print(f"📚 API 문서: http://localhost:8000/docs")
    print("=" * 60 + "\n")

# ⭐ 이제 모듈 import (여기서 GT 로드 메시지 출력됨)
from app.db.base import Base
from app.db.session import engine

# 라우터 import
from app.routers import swingRouters, uploadRouters, calendarRouters, gripRouters
from app.routers.reportRouters import router as report_router
from app.routers.userRouters import router as user_router

# 모델 import
from app.models.userModels import User
from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport


# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# FastAPI 앱 생성
app = FastAPI(
    title="MINTON-ANGLE API",
    description="배드민턴 자세 분석 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(user_router)
app.include_router(swingRouters.router)
app.include_router(uploadRouters.router)
app.include_router(calendarRouters.router)
app.include_router(report_router)
app.include_router(gripRouters.router)

# 정적 파일 서빙
app.mount("/data", StaticFiles(directory="/app/data"), name="data")


@app.get("/")
def read_root():
    return {"message": "MINTON-ANGLE API Server"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )