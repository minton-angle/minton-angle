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

# --- 3. 핵심 알고리즘 ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path): return None
    df = pd.read_csv(input_csv_path)
    
    # 데이터 정렬 및 인덱스 초기화
    df = df.sort_values('frame').reset_index(drop=True) if 'frame' in df.columns else df.reset_index(drop=True)

    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')}

    # 1. 기준점: 손목 최고점 (y가 0에 가장 가까운 순간)
    # 0에 가장 가까운 값을 찾기 위해 절댓값을 이용하거나 단순히 idxmin() 사용
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY (기존 알고리즘 유지)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(1.0)
    r_idx = df.iloc[:highest_idx]['wrist_move'].idxmin() if highest_idx > 0 else 0

    # 3. BACKSWING (기존 알고리즘 유지)
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_cands = bs_range[(bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.05)].copy()
    b_idx = bs_cands.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin() if not bs_cands.empty else bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    # 4. IMPACT: 최고점(highest_idx) 이후에서 추출하되 튀는 값 무시
    after_highest = df.iloc[highest_idx:].copy()
    
    if not after_highest.empty:
        # [튀는 값 무시 로직] 프레임 간 이동 거리가 0.15(화면의 15%) 이상이면 점프한 것으로 간주하여 제외
        after_highest['jump_dist'] = np.sqrt(after_highest[cols['wri'][0]].diff()**2 + 
                                             after_highest[cols['wri'][1]].diff()**2).fillna(0)
        
        # 정상적인 범미(이동 거리 0.15 미만)의 데이터만 필터링
        clean_impact_df = after_highest[after_highest['jump_dist'] < 0.15].copy()
        
        if not clean_impact_df.empty:
            # 필터링된 데이터에서 각도 계산
            clean_impact_df['temp_angle'] = clean_impact_df.apply(
                lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
                                          r[cols['wri'][0]], r[cols['wri'][1]],
                                          r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)

            # 손목이 팔꿈치보다 위(y+0.05 여유)에 있는 지점 중 160.5도와 가장 가까운 지점
            impact_window = clean_impact_df[clean_impact_df[cols['wri'][1]] <= clean_impact_df[cols['elb'][1]] + 0.05].copy()
            
            if not impact_window.empty:
                i_idx = abs(impact_window['temp_angle'] - 160.5).idxmin()
            else:
                i_idx = clean_impact_df['temp_angle'].idxmax()
        else:
            i_idx = highest_idx
    else:
        i_idx = highest_idx

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}
    print(f"분석 완료: 최고점(y≈0) 이후 임팩트 탐색 및 노이즈 제외 적용 -> {result}")

    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    if output_img_dir: generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    return result

if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/GT4_normalized_fixed.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_keyframes_8.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_visualized_frames_8"
    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)