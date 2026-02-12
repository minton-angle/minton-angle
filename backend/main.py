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

from app.db.base import Base
from app.db.session import engine

# 라우터 import
from app.routers import swingRouters, uploadRouters
from app.routers import swingRouters, uploadRouters, calendarRouters

# 모델 import (테이블 생성용)
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

# # CORS 설정 (더 구체적으로)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5500",      # Live Server
#         "http://127.0.0.1:5500",      # Live Server
#         "http://localhost:8000",      # 백엔드 자체
#         "http://127.0.0.1:8000",
#         "null"                        # 파일:// 프로토콜
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 라우터 등록
app.include_router(swingRouters.router)
app.include_router(uploadRouters.router)
app.include_router(calendarRouters.router)

# data 폴더를 정적 파일로 제공
app.mount("/data", StaticFiles(directory="data"), name="data")

# 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "MINTON-ANGLE API Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 MINTON-ANGLE API Server 시작")
    print("=" * 60)
    print(f"📁 작업 디렉토리: {current_dir}")
    print(f"🌐 서버 주소: http://localhost:8000")
    print(f"📚 API 문서: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )