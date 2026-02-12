import pandas as pd
import numpy as np
import cv2
import os

# --- 1. 설정: 경로 및 포인트 ---
CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_1_data.csv'  # 아까 저장한 CSV 파일 경로
OUTPUT_VIDEO = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_1_skeleton.mp4'  # 저장할 영상 이름
WIDTH, HEIGHT = 1280, 720              # 캔버스 크기 (원본 영상 해상도에 맞추면 좋습니다)
FPS = 24                               # 영상 재생 속도

# 민지의 19개 핵심 포인트
SELECTED_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# 19개 포인트 내에서 연결할 뼈대 쌍 (번호 기반)
# 관절들을 선으로 이어서 사람 형태로 보이게 합니다.
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # 상체 및 팔
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), # 손 (추가 포인트 포함 시)
    (11, 23), (12, 24), (23, 24), # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28), # 다리
    (27, 29), (29, 31), (27, 31), # 왼발
    (28, 30), (30, 32), (28, 32)  # 오른발
]

# --- 2. 데이터 불러오기 ---
df = pd.read_csv(CSV_PATH)

# 영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

print(f"🎨 시각화 시작: {CSV_PATH} 데이터를 검은 배경에 그립니다.")

# --- 3. 프레임별 그리기 작업 ---
for i, row in df.iterrows():
    # 검은색 배경 생성 (0으로 채워진 행렬)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # 관절 위치 저장용 딕셔너리 (선 그릴 때 사용)
    points = {}

    # 1. 점(Joints) 그리기
    for idx in SELECTED_INDICES:
        # CSV 컬럼 이름이 'idx_x', 'idx_y' 형식일 경우
        x = row[f'{idx}_x']
        y = row[f'{idx}_y']
        v = row[f'{idx}_v'] # 가시성
        
        # 좌표가 존재하고(0이 아니고) 가시성이 일정 수준 이상일 때만 그림
        if x != 0 and y != 0 and v > 0.5:
            cx, cy = int(x * WIDTH), int(y * HEIGHT)
            points[idx] = (cx, cy)
            
            # 노란색 원 그리기 (민지의 19개 포인트 강조)
            cv2.circle(canvas, (cx, cy), 6, (0, 255, 255), -1)
            # 번호 표시 (선택 사항)
            # cv2.putText(canvas, str(idx), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 2. 선(Skeleton) 그리기
    for start, end in SKELETON_CONNECTIONS:
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], (0, 255, 0), 2) # 초록색 선

    # 3. 프레임 정보 표시
    cv2.putText(canvas, f"Frame: {int(row['frame'])}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 결과물 보여주기 및 저장
    out.write(canvas)
    cv2.imshow('Black Background Skeleton', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
print(f"🎉 시각화 완료! 결과 영상: {os.path.abspath(OUTPUT_VIDEO)}")