import pandas as pd
import numpy as np
import cv2
import os

# ==========================================
# 1. 데이터 처리 함수 (CSV 생성)
# ==========================================
def calculate_badminton_averages_with_details(norm_files, kf_files, output_path):
    stages = ['ready', 'backswing', 'impact']
    all_rows = [] 

    for stage in stages:
        stage_data = [] 
        for i, (norm_path, kf_path) in enumerate(zip(norm_files, kf_files), 1):
            try:
                norm_df = pd.read_csv(norm_path)
                kf_df = pd.read_csv(kf_path)
                
                if stage in kf_df.columns:
                    target_frame = kf_df[stage].iloc[0]
                    row = norm_df[norm_df['frame_id'] == target_frame].copy()
                    
                    if not row.empty:
                        row = row.drop(columns=['frame_id', 'timestamp'])
                        row['stage'] = stage
                        row['source'] = f'GT{i}'
                        stage_data.append(row)
            except Exception as e:
                print(f"❌ {norm_path} 처리 중 오류 발생: {e}")

        if stage_data:
            stage_df = pd.concat(stage_data)
            # 평균 계산
            avg_values = stage_df.mean(numeric_only=True).to_frame().T
            avg_values['stage'] = stage
            avg_values['source'] = 'Average'
            
            all_rows.append(stage_df)
            all_rows.append(avg_values)

    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        final_df = final_df.round(4) # 소수점 4자리 제한
        
        # 컬럼 순서 조정
        cols = ['stage', 'source'] + [c for c in final_df.columns if c not in ['stage', 'source']]
        final_df = final_df[cols]
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 생성 완료: {output_path}")
        return final_df
    else:
        print("❌ 추출된 데이터가 없습니다.")
        return None

# ==========================================
# 2. 시각화 함수 (이미지 생성)
# ==========================================
def visualize_average_poses(csv_path, output_dir, img_size=(400, 600), padding=0.15):
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 'Average' 소스 데이터만 추출
    avg_df = df[df['source'] == 'Average']
    
    W, H = img_size
    draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
    off_x, off_y = W * padding, H * padding

    SKELETON_CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_wrist', 'left_pinky'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_wrist', 'right_pinky'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot_index'), ('left_heel', 'left_foot_index'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot_index'), ('right_heel', 'right_foot_index'),
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear')
    ]

    for _, row in avg_df.iterrows():
        stage = row['stage']
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        pts_px = {}

        for col in df.columns:
            if col.endswith('_x'):
                base_name = col[:-2]
                y_col = base_name + '_y'
                if y_col in df.columns:
                    x_val, y_val = row[col], row[y_col]
                    if not pd.isna(x_val) and not pd.isna(y_val):
                        px = int(off_x + x_val * draw_w)
                        py = int(off_y + y_val * draw_h)
                        pts_px[base_name] = (px, py)
                        color = (0, 0, 255) if 'nose' in base_name else (0, 255, 0)
                        cv2.circle(canvas, (px, py), 5, color, -1)

        for start_pt, end_pt in SKELETON_CONNECTIONS:
            if start_pt in pts_px and end_pt in pts_px:
                cv2.line(canvas, pts_px[start_pt], pts_px[end_pt], (255, 255, 255), 2)

        text = f"AVG {stage.upper()}"
        cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_path = os.path.join(output_dir, f"avg_pose_{stage}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"📸 이미지 저장 완료: {save_path}")

# ==========================================
# 3. 통합 실행부
# ==========================================
if __name__ == "__main__":
    # --- [경로 설정] ---
    normalized_csv_list = [
        '/Users/minji/Documents/minton-angle_resources/GT1_normalized_fixed.csv', 
        '/Users/minji/Documents/minton-angle_resources/GT2_normalized_fixed.csv', 
        '/Users/minji/Documents/minton-angle_resources/GT3_normalized_fixed.csv', 
        '/Users/minji/Documents/minton-angle_resources/GT4_normalized_fixed.csv'
    ]
    keyframe_csv_list = [
        '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv', 
        '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT2.csv', 
        '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT3.csv', 
        '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT4.csv'
    ]
    
    SAVE_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyframes_detailed_2.csv'
    OUTPUT_IMAGE_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyframe_images_2'

    # --- [실행] ---
    # 1. 상세 CSV 생성 및 평균 계산
    result_df = calculate_badminton_averages_with_details(normalized_csv_list, keyframe_csv_list, SAVE_CSV_PATH)

    # 2. 생성된 CSV를 바탕으로 시각화 수행
    if result_df is not None:
        visualize_average_poses(SAVE_CSV_PATH, OUTPUT_IMAGE_DIR)