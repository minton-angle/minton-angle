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
    
    # 분모가 0이 되는 것을 방지
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# --- 2. 시각화 함수: 모든 키포인트 전신 스켈레톤 ---
def generate_keyframe_images_full(df, results_dict, output_dir, base_filename, img_size=(400, 640)):
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size

    SKELETON_CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('nose', 'left_eye'), ('nose', 'right_eye')
    ]

    print(f"\n[시각화 시작] 폴더: {output_dir}")

    for label, frame_idx in results_dict.items():
        if frame_idx < 0 or frame_idx >= len(df): continue
        frame_data = df.iloc[frame_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        points_px = {}

        # 모든 점 그리기
        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                y_col = base + '_y'
                if y_col in df.columns and not pd.isna(frame_data[col]):
                    px, py = int(frame_data[col] * W), int(frame_data[y_col] * H)
                    points_px[base] = (px, py)
                    color = (0, 0, 255) if 'nose' in base else (0, 255, 0)
                    cv2.circle(canvas, (px, py), 5, color, -1)

        # 선 연결
        for start, end in SKELETON_CONNECTIONS:
            if start in points_px and end in points_px:
                cv2.line(canvas, points_px[start], points_px[end], (255, 255, 255), 2)

        # 정보 기입
        cv2.putText(canvas, f"{label.upper()} F:{frame_idx}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        save_path = os.path.join(output_dir, f"{base_filename}_{label}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f" - {label} 저장 완료: {save_path}")

# --- 3. 핵심 알고리즘 함수 (제공해주신 코드 로직 반영) ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} 파일을 찾을 수 없습니다."); return None

    df = pd.read_csv(input_csv_path)
    
    # [컬럼 설정] CSV 파일의 실제 컬럼명에 맞게 확인 필요
    cols = {
        'nose': ('nose_x', 'nose_y'),
        'shld': ('right_shoulder_x', 'right_shoulder_y'),
        'elb': ('right_elbow_x', 'right_elbow_y'),
        'wri': ('right_wrist_x', 'right_wrist_y'),
        'hand': ('right_pinky_x', 'right_pinky_y') # 17, 18번 평균값이 있다면 그 컬럼 사용
    }

    # 0. 필수 데이터 계산 (스냅 각도 및 손목 이동량)
    df['snap_angle'] = df.apply(lambda r: calculate_angle(
        r[cols['elb'][0]], r[cols['elb'][1]],
        r[cols['wri'][0]], r[cols['wri'][1]],
        r[cols['hand'][0]], r[cols['hand'][1]]
    ), axis=1)

    # 손목의 프레임 간 이동량 (Movement)
    df['wrist_move'] = np.sqrt(df[cols['wri'][0]].diff()**2 + df[cols['wri'][1]].diff()**2).fillna(1.0)

    # 1. 기준점: 손목 최고점 (Y값 최소 지점)
    highest_idx = df[cols['wri'][1]].idxmin()

    # 2. READY 추출
    # 최고점 이전 구간 중 손목 움직임이 가장 적은(멈칫하는) 지점
    before_highest = df.iloc[:highest_idx]
    if not before_highest.empty:
        r_idx = before_highest['wrist_move'].idxmin()
    else:
        r_idx = 0

    # 3. IMPACT 추출
    # 최고점 이후 + 손목이 팔꿈치보다 위(y가 작음) + 스냅 각도가 160.5도에 가장 가까운 지점
    impact_candidates = df[(df.index >= highest_idx) & (df[cols['wri'][1]] <= df[cols['elb'][1]])].copy()
    if not impact_candidates.empty:
        # 160.5도와의 차이 계산
        impact_candidates['angle_diff'] = abs(impact_candidates['snap_angle'] - 160.5)
        i_idx = impact_candidates['angle_diff'].idxmin()
    else:
        i_idx = highest_idx

    # 4. BACKSWING 추출
    # Ready와 최고점 사이 + 손목이 코보다 뒤(x가 작음) + 어깨보다 팔꿈치가 들려있음
    # 그 중 손목과 팔꿈치의 x좌표 차이가 최소인 지점
    bs_range = df.iloc[r_idx:highest_idx+1].copy()
    bs_candidates = bs_range[
        (bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
        (bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]] + 0.05)
    ].copy()

    if not bs_candidates.empty:
        bs_candidates['x_diff'] = abs(bs_candidates[cols['wri'][0]] - bs_candidates[cols['elb'][0]])
        b_idx = bs_candidates['x_diff'].idxmin()
    else:
        # 조건 만족 없을 시 단순 x차이 최소 지점
        b_idx = bs_range.apply(lambda r: abs(r[cols['wri'][0]] - r[cols['elb'][0]]), axis=1).idxmin()

    result = {'ready': int(r_idx), 'backswing': int(b_idx), 'impact': int(i_idx)}

    # 결과 출력 및 저장
    print(f"\n--- 분석 결과 ({os.path.basename(input_csv_path)}) ---")
    print(f"READY: {result['ready']}, BACKSWING: {result['backswing']}, IMPACT: {result['impact']}")

    if output_csv_path:
        pd.DataFrame([result]).to_csv(output_csv_path, index=False)

    if output_img_dir:
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
        generate_keyframe_images_full(df, result, output_img_dir, base_name)

    return result

# --- 실행부 ---
if __name__ == "__main__":
    # 사용자 환경에 맞춘 경로 설정
    INPUT_CSV = "/Users/minji/Documents/GT1_normalized.csv"
    RESULT_CSV = "/Users/minji/Documents/minton-angle/backend/data/standard/GT1_norm_keyframes_2.csv"
    IMAGE_DIR = "/Users/minji/Documents/minton-angle/backend/data/standard/GT1_norm_visualized_frames_2"

    analyze_swing_keyframes(INPUT_CSV, RESULT_CSV, IMAGE_DIR)