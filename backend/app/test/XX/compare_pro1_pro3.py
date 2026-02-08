import pandas as pd
import numpy as np
import cv2
import os

# --- 1. 경로 및 설정 (여기만 수정하세요) ---
GT_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_1_data.csv'      # 정석(GT) 데이터 경로
USER_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/pro_swing_3_data.csv'    # 사용자 데이터 경로
OUTPUT_VIDEO_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/comparison_result.mp4' # 저장할 결과 경로

# 결과 영상 설정
WIDTH, HEIGHT = 640, 480  # 개별 화면 크기 (합치면 1280x480 + 점수판 100)
FPS = 30

# 민지의 19개 핵심 포인트 인덱스
SELECTED_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# 뼈대 연결 정보 (19개 내에서 연결)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # 상체
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), # 하체
    (27, 29), (29, 31), (28, 30), (30, 32) # 발
]

# --- 2. 수학적 계산 함수 ---
def calculate_angle(p1, p2, p3):
    """세 점 사이의 내각을 계산 (p2가 꼭짓점)"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def draw_styled_skeleton(img, row, offset_x=0):
    """검정 배경에 뼈대를 그리는 함수"""
    points = {}
    for idx in SELECTED_INDICES:
        x, y = int(row[f'{idx}_x'] * WIDTH) + offset_x, int(row[f'{idx}_y'] * HEIGHT)
        points[idx] = (x, y)
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1) # 노란색 관절

    for start, end in CONNECTIONS:
        if start in points and end in points:
            cv2.line(img, points[start], points[end], (0, 255, 0), 2) # 초록색 뼈대

# --- 3. 실행 및 저장 로직 ---
# 폴더 생성
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

df_gt = pd.read_csv(GT_CSV_PATH)
df_user = pd.read_csv(USER_CSV_PATH)

# 영상 저장 객체
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (WIDTH * 2, HEIGHT + 120))

min_frames = min(len(df_gt), len(df_user))

for i in range(min_frames):
    row_gt = df_gt.iloc[i]
    row_user = df_user.iloc[i]
    
    # 1280 x 600 크기의 검정색 캔버스 (영상 2개 + 하단 바)
    canvas = np.zeros((HEIGHT + 120, WIDTH * 2, 3), dtype=np.uint8)
    
    # 뼈대 그리기
    draw_styled_skeleton(canvas, row_gt, offset_x=0)
    draw_styled_skeleton(canvas, row_user, offset_x=WIDTH)

    # 주요 관절 각도 비교 (예: 팔꿈치)
    # 12-14-16 (오른쪽 팔 각도)
    gt_ang = calculate_angle([row_gt['12_x'], row_gt['12_y']], [row_gt['14_x'], row_gt['14_y']], [row_gt['16_x'], row_gt['16_y']])
    user_ang = calculate_angle([row_user['12_x'], row_user['12_y']], [row_user['14_x'], row_user['14_y']], [row_user['16_x'], row_user['16_y']])
    
    # 유사도 계산 (100 - 각도 차이)
    diff = abs(gt_ang - user_ang)
    similarity = max(0, 100 - diff)

    # 시각화 텍스트
    cv2.putText(canvas, f"PRO CLEAR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"USER POSE", (WIDTH + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 하단 점수판
    score_color = (0, 255, 0) if similarity > 80 else (0, 165, 255) # 80점 이상 초록색
    cv2.rectangle(canvas, (50, HEIGHT + 40), (int(50 + similarity * 10), HEIGHT + 90), score_color, -1)
    cv2.putText(canvas, f"Similarity: {similarity:.1f}%", (50, HEIGHT + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(canvas)
    cv2.imshow('Minton-Angle Analysis', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap_release = [out.release(), cv2.destroyAllWindows()]
print(f"✅ 비교 영상 저장 완료: {os.path.abspath(OUTPUT_VIDEO_PATH)}")