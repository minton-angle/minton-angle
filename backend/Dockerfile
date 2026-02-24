# 1. 파이썬 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 2. 시스템 필수 라이브러리 설치 (OpenCV 구동을 위한 최신 패키지들)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]