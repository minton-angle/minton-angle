# 키 프레임 추출 알고리즘
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

# --- 2. 시각화 함수: 발 구조 강화 및 중앙 정렬 ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(400, 600), padding=0.15):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    # [스켈레톤 연결 정의] 발목-뒤꿈치-발끝을 연결하여 발 모양 형성
    SKELETON_CONNECTIONS = [
        # 상체 및 팔
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        # 다리 및 강화된 발 구조
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot_index'), ('left_heel', 'left_foot_index'), # 왼발 삼각형
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot_index'), ('right_heel', 'right_foot_index'), # 오른발 삼각형
        # 얼굴
        ('nose', 'left_eye'), ('nose', 'right_eye')
    ]

    for label, frame_idx in results_dict.items():
        if frame_idx < 0 or frame_idx >= len(df): continue
        frame_data = df.iloc[frame_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        pts_px = {}

        # 좌표 변환 및 점 그리기
        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                y_col = base + '_y'
                if y_col in df.columns and not pd.isna(frame_data[col]):
                    px = int(off_x + frame_data[col] * draw_w)
                    py = int(off_y + frame_data[y_col] * draw_h)
                    pts_px[base] = (px, py)
                    cv2.circle(canvas, (px, py), 5, (0, 0, 255) if 'nose' in base else (0, 255, 0), -1)

        # 선 연결
        for s, e in SKELETON_CONNECTIONS:
            if s in pts_px and e in pts_px:
                cv2.line(canvas, pts_px[s], pts_px[e], (255, 255, 255), 2)

        cv2.putText(canvas, f"{label.upper()} F:{frame_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_{label}.jpg"), canvas)

# --- 3. 핵심 알고리즘 (Impact 우선순위 고수) ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path): return None
    df = pd.read_csv(input_csv_path)
    
    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')}

    df['snap_angle'] = df.apply(lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
        r[cols['wri'][0]], r[cols['wri'][1]], r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(1.0)

    # 1. 기준점: 손목 최고점
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY: 최고점 이전 중 [팔꿈치가 어깨보다 뒤에 있음] + [정지 상태]
    # '뒤'의 기준: x좌표가 어깨보다 작음 (오른손잡이가 오른쪽을 보고 서 있을 때 기준)
    ready_range = df.iloc[:highest_idx].copy()
    ready_cands = ready_range[ready_range[cols['elb'][0]] < ready_range[cols['shld'][0]]]

    if not ready_cands.empty:
        # 조건에 맞는 프레임 중 움직임이 가장 적은 순간 선택
        r_idx = ready_cands['wrist_move'].idxmin()
    else:
        # 조건에 맞는 프레임이 없다면 단순히 최고점 이전 중 움직임 최소 지점 선택
        r_idx = ready_range['wrist_move'].idxmin() if highest_idx > 0 else 0

    # 3. IMPACT: 반드시 최고점(highest_idx) 이후에서만 추출
    after_highest_df = df.iloc[highest_idx:].copy()
    impact_window = after_highest_df[after_highest_df[cols['wri'][1]] <= after_highest_df[cols['elb'][1]]].copy()
    
    if not impact_window.empty:
        impact_window['angle_diff'] = abs(impact_window['snap_angle'] - 160.5)
        i_idx = impact_window['angle_diff'].idxmin()
    else:
        i_idx = highest_idx

    # 4. BACKSWING: Ready와 최고점(highest_idx) 사이에서 추출
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    
    # [수정 로직] 사용자 요청 조건 반영
    # 1. 팔꿈치 X < 코 X (뒤쪽)
    # 2. 손목 X < 코 X (뒤쪽)
    # 3. 팔꿈치 Y < 코 Y (위쪽 - 좌표값이 작아야 위임)
    bs_cands = bs_range[
        (bs_range[cols['elb'][0]] < bs_range[cols['nose'][0]]) & 
        (bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
        (bs_range[cols['elb'][1]] < bs_range[cols['nose'][1]])
    ].copy()
    
    if not bs_cands.empty:
        # 조건을 만족하는 후보군 중 팔꿈치가 가장 높게 올라간 순간(Y 최소값)을 정점으로 판단
        b_idx = bs_cands[cols['elb'][1]].idxmin()
    else:
        # 만약 엄격한 조건을 만족하는 프레임이 없다면 (코보다 높이 안 올라갔을 경우)
        # 차선책으로 팔꿈치가 어깨보다 위에 있는 프레임 중 팔꿈치가 가장 높은 순간 선택
        fallback_cands = bs_range[bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]]].copy()
        if not fallback_cands.empty:
            b_idx = fallback_cands[cols['elb'][1]].idxmin()
        else:
            # 아예 후보가 없으면 Ready와 최고점의 중간 프레임 선택
            b_idx = int((r_idx + highest_idx) / 2)

    result = {
        'ready': int(r_idx),
        'backswing': int(b_idx),
        'impact': int(i_idx)
    }

    # ... (이하 동일)
    print(f"분석 및 시각화 완료: {result}")

    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    if output_img_dir: generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    return result

    # --- 실행부 ---
if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/minton-angle_resources/wooil_normalized_fixed.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/wooil.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/wooil"

    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)