import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

print(f"📁 작업 디렉토리: {current_dir}")
print(f"🔍 sys.path: {sys.path[:3]}")

try:
    import main
    print("✅ main 모듈 import 성공!")
except Exception as e:
    print(f"❌ main 모듈 import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )