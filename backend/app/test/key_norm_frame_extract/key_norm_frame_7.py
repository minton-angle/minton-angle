import pandas as pd
import numpy as np
import os
import cv2

# --- 1. 각도 계산 함수 ---
def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    a, b, c = np.array([a_x, a_y]), np.array([b_x, b_y]), np.array([c_x, c_y])
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)))

# --- 2. 시각화 함수: 발 구조 강화 (삼각형 연결) ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(400, 600), padding=0.15):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    SKELETON_CONNECTIONS = [
        # 상체 및 팔
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        # 다리 및 발 구조 (삼각형 완성)
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'), ('left_heel', 'left_foot_index'), ('left_foot_index', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ('right_heel', 'right_foot_index'), ('right_foot_index', 'right_ankle'),
        # 얼굴
        ('nose', 'left_eye'), ('nose', 'right_eye')
    ]

    for label, frame_idx in results_dict.items():
        if frame_idx < 0 or frame_idx >= len(df): continue
        frame_data = df.iloc[frame_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        pts_px = {}

        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                y_col = base + '_y'
                if y_col in df.columns and not pd.isna(frame_data[col]):
                    px = int(off_x + frame_data[col] * draw_w)
                    py = int(off_y + frame_data[y_col] * draw_h)
                    pts_px[base] = (px, py)
                    cv2.circle(canvas, (px, py), 5, (0, 0, 255) if 'nose' in base else (0, 255, 0), -1)

        for s, e in SKELETON_CONNECTIONS:
            if s in pts_px and e in pts_px:
                cv2.line(canvas, pts_px[s], pts_px[e], (255, 255, 255), 2)

        cv2.putText(canvas, f"{label.upper()} F:{frame_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_{label}.jpg"), canvas)

# --- 3. 핵심 알고리즘 (Impact 부분만 유연하게 수정) ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path): return None
    df = pd.read_csv(input_csv_path)
    
    # 에러 방지: 'frame' 컬럼 유무 확인 및 인덱스 초기화
    if 'frame' in df.columns:
        df = df.sort_values('frame').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')}

    # Raw 데이터 기반 물리량 계산 (기존 Ready/Backswing용)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(1.0)

    # 1. 기준점: 손목 최고점 (모든 분석의 기준)
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY (알고리즘 유지)
    r_idx = df.iloc[:highest_idx]['wrist_move'].idxmin() if highest_idx > 0 else 0

    # 3. IMPACT (이 부분만 최적화)
    # 최고점 이후 구간만 별도로 복사하여 로컬 노이즈 제거 진행
    after_highest_df = df.iloc[highest_idx : highest_idx + 50].copy() # 최고점 이후 50프레임 내외 탐색
    
    if not after_highest_df.empty:
        # Impact 판별용 로컬 스무딩 (팔 부위만)
        for part in ['right_wrist', 'right_elbow', 'right_pinky']:
            after_highest_df[f'{part}_x'] = after_highest_df[f'{part}_x'].rolling(window=5, center=True, min_periods=1).median()
            after_highest_df[f'{part}_y'] = after_highest_df[f'{part}_y'].rolling(window=5, center=True, min_periods=1).median()
        
        # 스무딩된 좌표로 각도 계산
        after_highest_df['temp_angle'] = after_highest_df.apply(
            lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
                                      r[cols['wri'][0]], r[cols['wri'][1]],
                                      r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)

        # 손목이 팔꿈치보다 위(y값이 작거나 같음)에 있는 프레임들 필터링 (+0.05 오차 허용으로 유연성 확보)
        impact_window = after_highest_df[after_highest_df[cols['wri'][1]] <= after_highest_df[cols['elb'][1]] + 0.05].copy()
        
        if not impact_window.empty:
            # 목표 각도(160.5도)에 가장 가까운 지점 탐색
            i_idx = abs(impact_window['temp_angle'] - 160.5).idxmin()
        else:
            # 적절한 지점이 없다면 구간 내에서 가장 팔이 많이 펴진 곳 선택
            i_idx = after_highest_df['temp_angle'].idxmax()
    else:
        i_idx = highest_idx

    # 4. BACKSWING (알고리즘 유지)
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_cands = bs_range[(bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.05)].copy()
    b_idx = bs_cands.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin() if not bs_cands.empty else bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}
    print(f"분석 완료: {result}")

    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    if output_img_dir: generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    return result

if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/GT4_normalized.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_keyframes_5.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_visualized_frames_5"

    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)