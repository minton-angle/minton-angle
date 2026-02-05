import pandas as pd
import numpy as np
import os
import cv2

# --- 1. 유틸리티 함수: 각도 계산 ---
def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    """세 점 a, b, c 사이의 각도를 계산 (b가 꼭짓점)"""
    a = np.array([a_x, a_y])
    b = np.array([b_x, b_y])
    c = np.array([c_x, c_y])
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- 2. 시각화 함수: 중앙 정렬 및 추가 연결 반영 ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(400, 600), padding=0.15):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size

    # [그리기 영역 계산] 중앙 정렬 및 여백을 위한 내부 영역 설정
    draw_w = W * (1 - 2 * padding)
    draw_h = H * (1 - 2 * padding)
    offset_x = W * padding
    offset_y = H * padding

    # [스켈레톤 연결 정의 업데이트] 발, 손 연결 추가
    # 주의: CSV 파일에 해당 컬럼(예: right_foot_index_x)이 있어야 선이 그려집니다.
    SKELETON_CONNECTIONS = [
        # 몸통
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        # 팔
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'), # 손 연결 추가
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'), # 손 연결 추가
        # 다리
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('left_ankle', 'left_foot_index'), # 발 연결 추가
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('right_ankle', 'right_foot_index'), # 발 연결 추가
        # 얼굴
        ('nose', 'left_eye'), ('nose', 'right_eye')
    ]

    print(f"\n[시각화 시작] 폴더: {output_dir}")

    for label, frame_idx in results_dict.items():
        if frame_idx < 0 or frame_idx >= len(df): continue
        frame_data = df.iloc[frame_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        points_px = {}

        # 모든 점 그리기 (좌표 변환에 여백 및 오프셋 적용)
        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                y_col = base + '_y'
                if y_col in df.columns and not pd.isna(frame_data[col]):
                    # 정규화 좌표(0~1)를 여백이 있는 내부 영역 좌표로 변환
                    norm_x, norm_y = frame_data[col], frame_data[y_col]
                    px = int(offset_x + norm_x * draw_w)
                    py = int(offset_y + norm_y * draw_h)
                    
                    points_px[base] = (px, py)
                    color = (0, 0, 255) if 'nose' in base else (0, 255, 0)
                    cv2.circle(canvas, (px, py), 5, color, -1)

        # 선 연결
        for start, end in SKELETON_CONNECTIONS:
            if start in points_px and end in points_px:
                cv2.line(canvas, points_px[start], points_px[end], (255, 255, 255), 2)

        # 정보 기입 (글자 크기 약간 조절)
        cv2.putText(canvas, f"{label.upper()} F:{frame_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_path = os.path.join(output_dir, f"{base_filename}_{label}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f" - {label} 저장 완료")

# --- 3. 핵심 알고리즘 함수 (이전과 동일) ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} 파일을 찾을 수 없습니다."); return None
    df = pd.read_csv(input_csv_path)
    
    # [분석용 컬럼] 실제 CSV 컬럼명 확인 필요
    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')} # right_pinky나 right_index 사용 권장

    # 0. 데이터 계산
    df['snap_angle'] = df.apply(lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
        r[cols['wri'][0]], r[cols['wri'][1]], r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(1.0)

    # 1. 기준점: 손목 최고점
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY
    before_highest = df.iloc[:highest_idx]
    r_idx = before_highest['wrist_move'].idxmin() if not before_highest.empty else 0

    # 3. IMPACT (스냅 각도 160.5도 기준)
    impact_cands = df[(df.index >= highest_idx) & (df[cols['wri'][1]] <= df[cols['elb'][1]])].copy()
    if not impact_cands.empty:
        impact_cands['angle_diff'] = abs(impact_cands['snap_angle'] - 160.5)
        i_idx = impact_cands['angle_diff'].idxmin()
    else: i_idx = highest_idx

    # 4. BACKSWING (손목-팔꿈치 x거리 최소)
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_cands = bs_range[(bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.05)].copy()
    if not bs_cands.empty:
        b_idx = bs_cands.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()
    else:
        b_idx = bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}
    print(f"\n--- 분석 결과 --- {result}")

    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    # 이미지 생성 함수 호출 시 크기 지정 가능 (기본값 400x600 사용)
    if output_img_dir: generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    return result

# --- 실행부 ---
if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/GT4_normalized.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_keyframes.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_visualized_frames"

    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)