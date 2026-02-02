from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MINTON-ANGLE FastAPI")

# 프론트엔드와 통신하기 위한 CORS 설정 =
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "MINTON-ANGLE 서버가 정상적으로 작동 중입니다"}

# 나중에 여기에 routers를 연결할 예정
# app.include_router(analysis.router)