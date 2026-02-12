import pandas as pd
import numpy as np
import os
import cv2

# --- 1. 시각화 함수 (모든 키포인트 동적 시각화로 수정됨) ---
def generate_keyframe_images_all_points(df, results_dict, output_dir, base_filename, img_size=(640, 640)):
    """
    CSV에 존재하는 모든 키포인트를 검은 배경에 동적으로 그려 이미지로 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    W, H = img_size

    # [표준 골격 연결 정의]
    # CSV의 컬럼명 베이스(예: left_shoulder_x 에서 '_x'를 뺀 부분)와 일치해야 선이 그려집니다.
    # 만약 선이 안 그려진다면 CSV 컬럼명과 이 리스트의 이름이 일치하는지 확인이 필요합니다.
    SKELETON_CONNECTIONS = [
        # 상체
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        # 팔
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        # 다리
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        # 얼굴 (코와 눈/귀 연결 - 데이터에 따라 다를 수 있음)
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear')
    ]

    print(f"\n[전신 이미지 생성 중...] 저장 위치: {output_dir}")

    for label, frame_idx in results_dict.items():
        if frame_idx >= len(df): continue
        frame_data = df.iloc[frame_idx]

        # 검은 배경 캔버스 생성
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 현재 프레임에서 그릴 수 있는 점들의 픽셀 좌표 저장소
        available_points_px = {} 

        # --- A. 모든 키포인트 점 찍기 ---
        # 데이터프레임의 모든 컬럼을 순회하며 _x 로 끝나는 컬럼을 찾음
        for col_name in frame_data.index:
            if col_name.endswith('_x'):
                base_name = col_name[:-2] # 예: 'right_elbow_x' -> 'right_elbow'
                y_col_name = base_name + '_y'

                # 짝이 맞는 y 컬럼이 있고, 둘 다 데이터가 유효한 경우(NaN이 아님)
                if y_col_name in frame_data and not (pd.isna(frame_data[col_name]) or pd.isna(frame_data[y_col_name])):
                    norm_x = frame_data[col_name]
                    norm_y = frame_data[y_col_name]

                    # 정규화 좌표 -> 픽셀 좌표 변환
                    px, py = int(norm_x * W), int(norm_y * H)
                    
                    # 나중에 선을 그리기 위해 좌표 저장
                    available_points_px[base_name] = (px, py)
                    
                    # 점 그리기 (코는 빨강, 나머지는 초록)
                    color = (0, 0, 255) if 'nose' in base_name else (0, 255, 0)
                    # 약간 작게 그림 (반지름 5)
                    cv2.circle(canvas, (px, py), 5, color, -1)

        # --- B. 골격 선 그리기 ---
        # 정의된 연결 리스트를 순회하며 양쪽 점이 모두 존재하면 선을 그림
        for start_part, end_part in SKELETON_CONNECTIONS:
            if start_part in available_points_px and end_part in available_points_px:
                start_pt = available_points_px[start_part]
                end_pt = available_points_px[end_part]
                # 흰색 선 그리기
                cv2.line(canvas, start_pt, end_pt, (255, 255, 255), 2)

        # --- C. 텍스트 정보 삽입 ---
        text = f"{label.upper()} (Frame: {frame_idx})"
        cv2.putText(canvas, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- D. 파일 저장 ---
        save_path = os.path.join(output_dir, f"{base_filename}_{label}_full.jpg")
        cv2.imwrite(save_path, canvas)
        print(f" - {label} 전신 이미지 저장 완료")


# --- 2. 분석 함수 (이전과 동일, 이미지 함수 호출명만 변경) ---
def analyze_swing_keyframes(input_csv_path, output_csv_path=None, output_img_dir=None):
    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} 파일을 찾을 수 없습니다.")
        return None

    df = pd.read_csv(input_csv_path)
    
    # [분석용] 컬럼명 설정 (MediaPipe/COCO 포맷에 맞춰 확인 필요)
    nose = {'x': 'nose_x', 'y': 'nose_y'}
    elbow = {'x': 'right_elbow_x', 'y': 'right_elbow_y'}
    wrist = {'x': 'right_wrist_x', 'y': 'right_wrist_y'}
    # 손가락 끝이 있다면 그것을 사용, 없다면 wrist를 사용해도 됨
    hand_x_col = 'right_pinky_x' if 'right_pinky_x' in df.columns else 'right_wrist_x'
    hand_y_col = 'right_pinky_y' if 'right_pinky_y' in df.columns else 'right_wrist_y'
    hand = {'x': hand_x_col, 'y': hand_y_col}


    def get_dist(p1_x, p1_y, p2_x, p2_y):
        return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

    def get_velocity(col_x, col_y):
        return np.sqrt(df[col_x].diff()**2 + df[col_y].diff()**2).fillna(0)

    df['elbow_vel'] = get_velocity(elbow['x'], elbow['y'])
    df['wrist_vel'] = get_velocity(wrist['x'], wrist['y'])
    df['hand_vel'] = get_velocity(hand['x'], hand['y'])

    # --- READY ---
    min_elbow_x_idx = df[elbow['x']].idxmin()
    ready_window = df.iloc[max(0, min_elbow_x_idx-5) : min(len(df), min_elbow_x_idx+5)]
    ready_frame = ready_window['elbow_vel'].idxmin()

    # --- BACKSWING ---
    after_ready_df = df.iloc[ready_frame:]
    bs_candidates = after_ready_df[
        (after_ready_df[hand['x']] < after_ready_df[nose['x']]) &
        (after_ready_df[wrist['x']] < after_ready_df[nose['x']]) &
        (after_ready_df[elbow['x']] < after_ready_df[nose['x']])
    ].copy()

    if not bs_candidates.empty:
        bs_candidates['density'] = (
            get_dist(bs_candidates[hand['x']], bs_candidates[hand['y']], bs_candidates[wrist['x']], bs_candidates[wrist['y']]) +
            get_dist(bs_candidates[wrist['x']], bs_candidates[wrist['y']], bs_candidates[elbow['x']], bs_candidates[elbow['y']])
        )
        backswing_frame = bs_candidates['density'].idxmin()
    else:
        backswing_frame = ready_frame + 10 

    # --- IMPACT ---
    after_bs_df = df.iloc[backswing_frame:]
    wrist_highest_idx = after_bs_df[wrist['y']].idxmin()
    impact_window = df[(df.index > wrist_highest_idx) & (df[wrist['y']] < df[elbow['y']])].copy()

    if not impact_window.empty:
        impact_window['impact_score'] = impact_window['hand_vel'] / (impact_window['wrist_vel'] + 0.001)
        impact_frame = impact_window['impact_score'].idxmax()
    else:
        impact_frame = wrist_highest_idx

    result = {'ready': int(ready_frame), 'backswing': int(backswing_frame), 'impact': int(impact_frame)}

    # 결과 출력 및 CSV 저장
    print(f"\n--- 분석 결과 ({os.path.basename(input_csv_path)}) ---")
    for k, v in result.items(): print(f"{k:10}: Frame {v}")

    if output_csv_path:
        pd.DataFrame([result]).to_csv(output_csv_path, index=False)

    # [수정됨] 새로 만든 전신 시각화 함수 호출
    if output_img_dir:
        base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
        generate_keyframe_images_all_points(df, result, output_img_dir, base_filename)

    return result

# --- 3. 실행부 ---
if __name__ == "__main__":
    # 테스트용 더미 데이터 생성 (실제 실행시에는 주석 처리하고 본인의 CSV 경로를 입력하세요)
    # create_dummy_csv("test_skeleton.csv") 
    # input_path = "test_skeleton.csv"
    
    input_path = "/Users/minji/Documents/GT1_normalized.csv" # 실제 CSV 파일 경로
    output_path = "/Users/minji/Documents/minton-angle/backend/data/standard/GT1_norm_keyframes.csv"
    img_dir = "/Users/minji/Documents/minton-angle/backend/data/standard/GT1_norm_visualized_frames" # 이미지가 저장될 폴더명 변경
    
    analyze_swing_keyframes(input_path, output_path, output_img_dir=img_dir)