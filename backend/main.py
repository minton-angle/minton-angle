from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.base import Base
from app.db.session import engine

# ⭐ 수정: swing 라우터 import
from app.routers import swing

# 모든 모델 import (테이블 생성용)
from app.models.userModels import User
from app.models.postModels import Post
from app.models.fileModels import File
from app.models.analysisModels import Analysis
from app.models.llmReportModels import LLMReport

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

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

# ⭐ 수정: swing 라우터 등록
app.include_router(swing.router)

@app.get("/")
def read_root():
    return {"message": "MINTON-ANGLE API Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}