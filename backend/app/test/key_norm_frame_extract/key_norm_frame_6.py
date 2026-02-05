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

# --- 2. 시각화 함수 (원본 데이터 기준 시각화) ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(600, 800), padding=0.1):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    SKELETON_CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'), ('left_heel', 'left_foot_index'), ('left_foot_index', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ('right_heel', 'right_foot_index'), ('right_foot_index', 'right_ankle'),
        ('nose', 'left_eye'), ('nose', 'right_eye')
    ]

    for label, idx in results_dict.items():
        if idx not in df.index: continue
        frame_data = df.loc[idx]
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

        for s, e in SKELETON_CONNECTIONS:
            if s in pts_px and e in pts_px:
                cv2.line(canvas, pts_px[s], pts_px[e], (255, 255, 255), 2, cv2.LINE_AA)
        
        for base, (px, py) in pts_px.items():
            cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)

        cv2.putText(canvas, f"{label.upper()} (Index: {idx})", (20, 50), 1, 1.5, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_{label}.jpg"), canvas)

# --- 3. 핵심 알고리즘 ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path): return None
    df = pd.read_csv(input_csv_path)
    
    # 'frame' 컬럼 체크 및 정렬
    if 'frame_id' in df.columns:
        df = df.sort_values('frame_id').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')}

    # 공통 물리량 계산 (Raw Data 기준)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(0)

    # 1. 기준점: 손목 최고점 (Raw 데이터 사용)
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY: 최고점 이전 움직임 최소 지점
    r_idx = df.iloc[:highest_idx]['wrist_move'].idxmin() if highest_idx > 0 else 0

    # 3. IMPACT: 이 부분에서만 노이즈 제거 적용
    # 최고점 이후 약 40프레임 구간 추출
    impact_search_zone = df.iloc[highest_idx : highest_idx + 40].copy()
    
    if not impact_search_zone.empty:
        # [Local Smoothing] Impact 판별을 위한 좌표만 살짝 다듬기
        smooth_targets = ['right_wrist_x', 'right_wrist_y', 'right_elbow_x', 'right_elbow_y', 'right_pinky_x', 'right_pinky_y']
        for col in smooth_targets:
            impact_search_zone[col] = impact_search_zone[col].rolling(window=5, center=True, min_periods=1).median()
        
        # 다듬어진 좌표로 각도 다시 계산
        impact_search_zone['temp_angle'] = impact_search_zone.apply(
            lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
                                      r[cols['wri'][0]], r[cols['wri'][1]],
                                      r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)
        
        # 조건: 손목이 팔꿈치보다 위(y-0.05)에 있고, 목표 각도 160.5에 가장 가까운 지점
        impact_window = impact_search_zone[impact_search_zone[cols['wri'][1]] <= impact_search_zone[cols['elb'][1]] + 0.05].copy()
        
        if not impact_window.empty:
            i_idx = abs(impact_window['temp_angle'] - 160.5).idxmin()
        else:
            i_idx = impact_search_zone['temp_angle'].idxmax() # 조건 만족 못하면 가장 펴진 곳
    else:
        i_idx = highest_idx

    # 4. BACKSWING (Raw 데이터 기준)
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_cands = bs_range[(bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.1)]
    
    if not bs_cands.empty:
        b_idx = bs_cands.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()
    else:
        b_idx = bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}
    
    # 저장 및 시각화
    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    if output_img_dir:
        generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    
    print(f"분석 완료 (Impact 로컬 최적화 적용): {result}")
    return result

if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/GT4_normalized.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_keyframes_4.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_visualized_frames_4"
    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)