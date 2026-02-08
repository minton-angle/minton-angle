import pandas as pd
import numpy as np
import cv2
import os

def visualize_average_csv(csv_path, output_dir, img_size=(400, 600), padding=0.15):
    # 1. 데이터 로드
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, index_col='stage')
    os.makedirs(output_dir, exist_ok=True)
    
    W, H = img_size
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    # [사용자 정의 스켈레톤 연결]
    SKELETON_CONNECTIONS = [
        # 상체 및 팔
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        # 다리 및 강화된 발 구조
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot_index'), ('left_heel', 'left_foot_index'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot_index'), ('right_heel', 'right_foot_index'),
        # 얼굴
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear')
    ]

    # 2. 각 단계(stage)별 이미지 생성
    for stage, row in df.iterrows():
        # 검은색 캔버스 생성
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        pts_px = {}

        # 좌표 추출 및 픽셀 변환
        # CSV 컬럼명에서 '_x'로 끝나는 것들을 찾아 대응하는 '_y'와 쌍을 맺음
        for col in df.columns:
            if col.endswith('_x'):
                base_name = col[:-2]
                y_col = base_name + '_y'
                
                if y_col in df.columns:
                    x_val = row[col]
                    y_val = row[y_col]
                    
                    if not pd.isna(x_val) and not pd.isna(y_val):
                        # 정규화 좌표(0~1)를 캔버스 크기에 맞게 변환
                        px = int(off_x + x_val * draw_w)
                        py = int(off_y + y_val * draw_h)
                        pts_px[base_name] = (px, py)
                        
                        # 점 그리기 (코는 빨간색, 나머지는 초록색)
                        color = (0, 0, 255) if 'nose' in base_name else (0, 255, 0)
                        cv2.circle(canvas, (px, py), 5, color, -1)

        # 3. 선 연결 (스켈레톤)
        for start_pt, end_pt in SKELETON_CONNECTIONS:
            if start_pt in pts_px and end_pt in pts_px:
                cv2.line(canvas, pts_px[start_pt], pts_px[end_pt], (255, 255, 255), 2)

        # 4. 텍스트 추가 및 저장
        text = f"AVG {stage.upper()}"
        cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_path = os.path.join(output_dir, f"avg_pose_{stage}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"📸 이미지 저장 완료: {save_path}")

# --- 실행부 ---
if __name__ == "__main__":
    # 아까 생성한 평균 CSV 경로
    INPUT_AVG_CSV = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyframes.csv' 
    # 이미지가 저장될 폴더
    OUTPUT_IMAGE_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyfrmaes_images'
    
    visualize_average_csv(INPUT_AVG_CSV, OUTPUT_IMAGE_DIR)