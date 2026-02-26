"""
MINTON-ANGLE API Server - 최종 통합본
"""
import sys
import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# 현재 디렉토리 경로 설정
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

load_dotenv()

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 MINTON-ANGLE API Server 시작")
    print("=" * 60)
    print(f"📁 작업 디렉토리: {current_dir}")
    print("=" * 60 + "\n")

# 모듈 import
from app.db.base import Base
from app.db.session import engine
from app.routers import swingRouters, uploadRouters, calendarRouters, gripRouters
from app.routers.reportRouters import router as report_router
from app.routers.totalReportRouters import router as total_report_router
from app.routers.userRouters import router as user_router

# DB 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MINTON-ANGLE API",
    description="배드민턴 자세 분석 API",
    version="1.0.0"
)
@app.middleware("http")
async def add_ngrok_skip_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "any_value"
    return response
    
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
app.include_router(total_report_router)
app.include_router(gripRouters.router)

# 🚀 [핵심 수정] 정적 파일 경로 자동 설정
# 도커 환경(/app/data 존재 여부)인지 확인 후 경로 지정
if os.path.exists("/app/data"):
    # 배포 환경 (Docker)
    final_data_dir = "/app/data"
    print(f"🌐 [배포 모드] 정적 파일 경로: {final_data_dir}")
else:
    # 로컬 환경 (팀원 PC)
    final_data_dir = os.path.join(current_dir, "data")
    os.makedirs(final_data_dir, exist_ok=True)
    print(f"💻 [로컬 모드] 정적 파일 경로: {final_data_dir}")

app.mount("/data", StaticFiles(directory=final_data_dir), name="data")

@app.get("/")
def read_root():
    return {"message": "MINTON-ANGLE API Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)