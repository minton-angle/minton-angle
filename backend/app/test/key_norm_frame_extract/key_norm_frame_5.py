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

# --- 2. 시각화 함수: 발 구조 강화 및 가독성 개선 ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(600, 800), padding=0.1):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size
    # 중앙 정렬을 위한 스케일링 계산
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    SKELETON_CONNECTIONS = [
        # 상체 및 팔
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        # 다리 및 강화된 발 구조 (삼각형 완성)
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

        # 모든 좌표 추출
        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                y_col = base + '_y'
                if y_col in df.columns and not pd.isna(frame_data[col]):
                    # 정규화된 좌표를 캔버스 크기에 맞게 변환
                    px = int(off_x + frame_data[col] * draw_w)
                    py = int(off_y + frame_data[y_col] * draw_h)
                    pts_px[base] = (px, py)

        # 선 그리기 (연결성 강화)
        for s, e in SKELETON_CONNECTIONS:
            if s in pts_px and e in pts_px:
                cv2.line(canvas, pts_px[s], pts_px[e], (200, 200, 200), 2, cv2.LINE_AA)

        # 점 그리기
        for base, (px, py) in pts_px.items():
            color = (0, 0, 255) if 'nose' in base else (0, 255, 0)
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)

        # 라벨 텍스트
        cv2.putText(canvas, f"PHASE: {label.upper()}", (20, 40), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(canvas, f"FRAME: {frame_idx}", (20, 75), 1, 1.2, (150, 150, 150), 1)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_{label}.jpg"), canvas)

# --- 3. 핵심 알고리즘: 노이즈 저항력 강화 ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path): return None
    df = pd.read_csv(input_csv_path).sort_values('frame_id').reset_index(drop=True)
    
    # [노이즈 제거] 튀는 데이터를 잡기 위해 중간값 필터(Median Filter) 적용
    # 윈도우 사이즈 3~5 정도로 튀는 프레임의 영향을 최소화합니다.
    smooth_cols = ['right_wrist_x', 'right_wrist_y', 'right_elbow_x', 'right_elbow_y', 'right_pinky_x', 'right_pinky_y']
    for c in smooth_cols:
        df[c] = df[c].rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')

    cols = {'nose': ('nose_x', 'nose_y'), 'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'), 'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')}

    # 기본 물리량 계산
    df['snap_angle'] = df.apply(lambda r: calculate_angle(r[cols['elb'][0]], r[cols['elb'][1]],
        r[cols['wri'][0]], r[cols['wri'][1]], r[cols['hand'][0]], r[cols['hand'][1]]), axis=1)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(0)

    # 1. 기준점: 손목 최고점 (안정적인 분석을 위해 이동평균 사용)
    highest_idx = df[cols['wri'][1]].rolling(window=5, center=True).mean().idxmin()

    # 2. READY: 최고점 이전, 움직임이 가장 적은 구간
    r_idx = df.iloc[:highest_idx]['wrist_move'].idxmin() if highest_idx > 0 else 0

    # 3. IMPACT: 유연한 인식 로직
    # 최고점 이후, 팔꿈치보다 손목이 높고, 각도가 160.5도에 가장 근접한 지점
    after_highest_df = df.iloc[highest_idx:highest_idx+30].copy() # 최고점 이후 30프레임 내외 탐색
    
    if not after_highest_df.empty:
        # 조건 1: 손목이 팔꿈치보다 위(y값이 작음)
        # 조건 2: 데이터가 튀는 것을 방지하기 위해 이전 프레임과의 각도 변화량이 너무 크지 않은 지점 선호
        after_highest_df['angle_score'] = abs(after_highest_df['snap_angle'] - 160.5)
        
        # 팔꿈치 높이 제약을 약간 완화 (데이터 노이즈 대응)
        impact_cands = after_highest_df[after_highest_df[cols['wri'][1]] <= after_highest_df[cols['elb'][1]] + 0.02]
        
        if not impact_cands.empty:
            i_idx = impact_cands['angle_score'].idxmin()
        else:
            # 만약 위 조건에 맞는게 없다면 최고점 이후 각도가 가장 많이 펴진 곳 선택
            i_idx = after_highest_df['snap_angle'].idxmax()
    else:
        i_idx = highest_idx

    # 4. BACKSWING
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_cands = bs_range[(bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.05)]
    
    if not bs_cands.empty:
        b_idx = bs_cands.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()
    else:
        b_idx = bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}
    
    # 저장 및 시각화
    if output_csv_path: pd.DataFrame([result]).to_csv(output_csv_path, index=False)
    if output_img_dir: 
        generate_keyframe_images_full(df, result, output_img_dir, os.path.splitext(os.path.basename(input_csv_path))[0])
    
    print(f"분석 완료: {result}")
    return result

if __name__ == "__main__":
    INPUT_CSV = "/Users/minji/Documents/GT4_normalized.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_keyframes_3.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT4_norm_visualized_frames_3"

    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)