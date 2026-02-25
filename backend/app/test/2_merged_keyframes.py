import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==============================================================================
# [1] 설정 및 컬럼 매핑 (사용자 환경에 맞게 수정)
# ==============================================================================

INPUT_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/junseo/junseo.csv'
OUTPUT_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/junseo/junseo_backswing_no_result_1'
OUTPUT_CSV_NAME = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/junseo/junseo_backswing_no_results_1.csv'

# [매핑 수정 가이드]
# 오른쪽(Value)을 실제 CSV 컬럼명으로 수정하세요.
COLUMN_MAPPING = {
    'frame': 'frame_id',
    
    # --- 필수 좌표 (알고리즘 계산용) ---
    'nose_x': 'nose_x', 'nose_y': 'nose_y',
    'r_sh_x': 'right_shoulder_x', 'r_sh_y': 'right_shoulder_y',
    'r_el_x': 'right_elbow_x',    'r_el_y': 'right_elbow_y',
    'r_wr_x': 'right_wrist_x',    'r_wr_y': 'right_wrist_y',
    'r_wr_z': 'right_wrist_z',    # Z좌표 (없으면 무시됨)
    
    # --- 시각화용 (몸통 & 다리) ---
    'l_sh_x': 'left_shoulder_x', 'l_sh_y': 'left_shoulder_y',
    'l_el_x': 'left_elbow_x',    'l_el_y': 'left_elbow_y',
    'l_wr_x': 'left_wrist_x',    'l_wr_y': 'left_wrist_y',
    'r_hip_x': 'right_hip_x',    'r_hip_y': 'right_hip_y',
    'l_hip_x': 'left_hip_x',     'l_hip_y': 'left_hip_y',
    'r_knee_x': 'right_knee_x',  'r_knee_y': 'right_knee_y',
    'l_knee_x': 'left_knee_x',   'l_knee_y': 'left_knee_y',
    'r_ank_x': 'right_ankle_x',  'r_ank_y': 'right_ankle_y',
    'l_ank_x': 'left_ankle_x',   'l_ank_y': 'left_ankle_y',
    
    # --- 발 디테일 (Heel & Foot Index) ---
    'r_heel_x': 'right_heel_x',       'r_heel_y': 'right_heel_y',
    'r_foot_idx_x': 'right_foot_index_x', 'r_foot_idx_y': 'right_foot_index_y',
    'l_heel_x': 'left_heel_x',        'l_heel_y': 'left_heel_y',
    'l_foot_idx_x': 'left_foot_index_x',  'l_foot_idx_y': 'left_foot_index_y',

    # 손끝 (시각화용)
    'r_fin_x': 'right_index_x',  'r_fin_y': 'right_index_y'
}

# 뼈대 연결 정보
SKELETON_CONNECTIONS = [
    # 상체
    ('nose', 'r_sh'), ('r_sh', 'r_el'), ('r_el', 'r_wr'), ('r_wr', 'r_fin'),
    ('nose', 'l_sh'), ('l_sh', 'l_el'), ('l_el', 'l_wr'),
    ('r_sh', 'l_sh'), ('r_sh', 'r_hip'), ('l_sh', 'l_hip'), ('r_hip', 'l_hip'),
    # 하체
    ('r_hip', 'r_knee'), ('r_knee', 'r_ank'),
    ('l_hip', 'l_knee'), ('l_knee', 'l_ank'),
    # 발 (삼각형 연결)
    ('r_ank', 'r_heel'), ('r_heel', 'r_foot_idx'), ('r_ank', 'r_foot_idx'),
    ('l_ank', 'l_heel'), ('l_heel', 'l_foot_idx'), ('l_ank', 'l_foot_idx')
]

# ==============================================================================

def visualize_all_keypoints(row, keyframe_name, save_dir):
    """모든 키포인트 시각화"""
    plt.figure(figsize=(6, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # 1. 모든 점 찍기
    for col in row.index:
        if str(col).endswith('_x'):
            base_name = str(col)[:-2]
            y_col = base_name + '_y'
            if y_col in row.index:
                x, y = row[col], row[y_col]
                if not (np.isnan(x) or np.isnan(y) or x == 0 or y == 0):
                    ax.scatter(x, y, color='white', s=20, alpha=0.6)

    # 2. 선 그리기
    for start, end in SKELETON_CONNECTIONS:
        sx_c, sy_c = COLUMN_MAPPING.get(f"{start}_x"), COLUMN_MAPPING.get(f"{start}_y")
        ex_c, ey_c = COLUMN_MAPPING.get(f"{end}_x"), COLUMN_MAPPING.get(f"{end}_y")
        
        if (sx_c in row.index) and (sy_c in row.index) and \
           (ex_c in row.index) and (ey_c in row.index):
            sx, sy = row[sx_c], row[sy_c]
            ex, ey = row[ex_c], row[ey_c]
            if not (np.isnan(sx) or np.isnan(sy) or np.isnan(ex) or np.isnan(ey)):
                ax.plot([sx, ex], [sy, ey], color='white', linewidth=2, alpha=0.8)

    frame_num = int(row[COLUMN_MAPPING['frame']]) if COLUMN_MAPPING['frame'] in row else '?'
    ax.set_title(f"{keyframe_name.upper()} (Frame {frame_num})", color='white')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    
    save_path = os.path.join(save_dir, f"{keyframe_name}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"이미지 저장: {save_path}")

def get_val(df, key):
    """매핑된 컬럼 데이터 가져오기"""
    col_name = COLUMN_MAPPING.get(key)
    if col_name in df.columns:
        return df[col_name]
    return None

def run_analysis():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: 파일을 찾을 수 없습니다 -> {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"CSV 로드 완료: {len(df)} frames")

    results = {}
    
    # 데이터 준비
    sh_x = get_val(df, 'r_sh_x') # 어깨 X
    el_x = get_val(df, 'r_el_x') # 팔꿈치 X
    wr_y = get_val(df, 'r_wr_y')
    wr_z = get_val(df, 'r_wr_z')
    
    # ---------------------------------------------------------
    # [1] READY (수정됨: 상대좌표 사용)
    # 조건: 오른쪽 어깨 기준, 팔꿈치가 가장 왼쪽(거리 차 최대)인 순간
    # 식: Maximize (Shoulder_X - Elbow_X)
    # ---------------------------------------------------------
    # 어깨보다 팔꿈치가 왼쪽에 있을 때(차이가 양수)만 고려하는 것이 안전함
    # 값이 클수록 팔꿈치가 어깨보다 더 왼쪽에 있다는 뜻
    relative_dist = sh_x - el_x 
    ready_idx = relative_dist.idxmax()
    
    results['ready'] = df.loc[ready_idx]
    print(f"[Ready] Frame {int(results['ready'][COLUMN_MAPPING['frame']])} (Rel. Dist: {relative_dist[ready_idx]:.2f})")

    # ---------------------------------------------------------
    # [2] 기준점 설정 (Highest Point)
    # ---------------------------------------------------------
    highest_idx = wr_y.idxmin()
    
    # ---------------------------------------------------------
    # [3] BACKSWING (엄격한 판단 로직 적용)
    # ---------------------------------------------------------
    start_search = ready_idx if ready_idx < highest_idx else 0
    bs_range = df.loc[start_search:highest_idx].copy()
    
    # [엄격한 조건]
    # 1. 팔꿈치(X)가 코(Nose)보다 뒤쪽(값은 작음)에 있어야 함 (오른손잡이 기준)
    # 2. 손목(X)도 코보다 뒤쪽에 있어야 함
    # 3. 팔꿈치(Y)가 어깨(Y)보다 높아야 함 (Y값은 작아야 함)
    
    # 방향에 따른 조건 설정 (왼쪽/오른쪽 보는 방향)
    nose_x = bs_range[COLUMN_MAPPING['nose_x']]
    el_x = bs_range[COLUMN_MAPPING['r_el_x']]
    wr_x = bs_range[COLUMN_MAPPING['r_wr_x']]
    el_y = bs_range[COLUMN_MAPPING['r_el_y']]
    sh_y = bs_range[COLUMN_MAPPING['r_sh_y']]

    # (방향 체크: 코가 어깨보다 왼쪽이면 -> 왼쪽을 보고 있음 -> 뒤쪽은 X가 커야 함)
    # (반대로 코가 어깨보다 오른쪽이면 -> 오른쪽을 보고 있음 -> 뒤쪽은 X가 작아야 함)
    # 여기서는 "오른쪽을 보고 타격한다"고 가정 (일반적 영상) -> 뒤쪽 = X가 작음
    
    c1 = el_x < nose_x  # 팔꿈치가 코 뒤
    c2 = wr_x < nose_x  # 손목이 코 뒤
    c3 = el_y < sh_y    # 팔꿈치가 어깨 위 (Y값은 위가 0이므로 작아야 높음)

    bs_cands = bs_range[c1 & c2 & c3]
    
    backswing_frame_val = None # 기본값: 없음

    if not bs_cands.empty:
        # 조건 만족하는 후보 중 팔꿈치가 가장 높이 올라간 순간(Y가 최소)
        backswing_idx = bs_cands[COLUMN_MAPPING['r_el_y']].idxmin()
        backswing_frame_val = int(df.loc[backswing_idx][COLUMN_MAPPING['frame']])
        results['backswing'] = df.loc[backswing_idx]
        print(f"[Backswing] Frame {backswing_frame_val}")
    else:
        print("[Backswing] ⚠️ 감지되지 않음 (조건 미충족)")
        # 억지로 impact를 넣지 않음!

    # ---------------------------------------------------------
    # [4] IMPACT (논리 기반 강화 버전 + 높이 계층 조건)
    # ---------------------------------------------------------
    # 1. 사람이 서 있는 방향 판별 (Ready 프레임 기준)
    facing_left = results['ready'][COLUMN_MAPPING['nose_x']] < results['ready'][COLUMN_MAPPING['r_sh_x']]

    # 2. 탐색 범위 설정 (최고점 근처 앞뒤)
    search_start = max(0, highest_idx - 10)
    search_end = min(len(df) - 1, highest_idx + 20)
    impact_search_range = df.loc[search_start:search_end].copy()

    best_impact_score = -1
    impact_idx = highest_idx

    for idx, row in impact_search_range.iterrows():
        # 각 좌표 추출
        rx, ry = row[COLUMN_MAPPING['r_sh_x']], row[COLUMN_MAPPING['r_sh_y']]
        ex, ey = row[COLUMN_MAPPING['r_el_x']], row[COLUMN_MAPPING['r_el_y']]
        wx, wy = row[COLUMN_MAPPING['r_wr_x']], row[COLUMN_MAPPING['r_wr_y']]

        # --- [필수 조건 1] 손목이 어깨보다 '앞쪽'에 있는가? ---
        if facing_left:
            is_in_front = wx < rx
        else:
            is_in_front = wx > rx
        
        if not is_in_front:
            continue

        # --- [필수 조건 2] 높이 계층: 어깨 < 팔꿈치 < 손목 순으로 높은가? ---
        # 이미지 좌표계: 위가 0이므로 y값은 어깨(큰값) > 팔꿈치 > 손목(작은값) 순서여야 함
        is_height_hierarchy_ok = (ry > ey) and (ey > wy)
        
        if not is_height_hierarchy_ok:
            continue

        # --- [점수 산출] ---
        # 요소 1: 어깨-손목 거리 (Maximize)
        distance = np.sqrt((rx - wx)**2 + (ry - wy)**2)
        
        # 요소 2: 손목 높이 (Maximize Height = Minimize Y)
        height_score = 1.0 - wy 
        
        # 요소 3: 팔꿈치 각도 (Maximize Straightness)
        ba = np.array([rx - ex, ry - ey])
        bc = np.array([wx - ex, wy - ey])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        angle_score = angle / 180.0

        # 통합 점수 (거리 40% + 높이 40% + 각도 20%)
        total_score = (distance * 0.4) + (height_score * 0.4) + (angle_score * 0.2)

        if total_score > best_impact_score:
            best_impact_score = total_score
            impact_idx = idx

    results['impact'] = df.loc[impact_idx]
    print(f"[Impact] Frame {int(results['impact'][COLUMN_MAPPING['frame']])} (Score: {best_impact_score:.4f})")

    # ---------------------------------------------------------
    # [5] FOLLOW-THROUGH
    # ---------------------------------------------------------
    follow_success = "X"
    follow_range = df.loc[impact_idx:]
    
    if wr_z is not None and not wr_z.isna().all():
        try:
            initial_z = wr_z.loc[impact_idx]
            initial_sign = np.sign(initial_z)
            for idx in follow_range.index:
                curr_z = wr_z.loc[idx]
                if pd.isna(curr_z): continue
                if np.sign(curr_z) != initial_sign and initial_sign != 0:
                    follow_success = "O"
                    break
        except Exception: pass

    if follow_success == "X":
        # 손목 X가 어깨 X보다 왼쪽(작음)
        fs_check = follow_range[follow_range[COLUMN_MAPPING['r_wr_x']] < follow_range[COLUMN_MAPPING['r_sh_x']]]
        if not fs_check.empty:
            follow_success = "O"

    print(f"[Follow-through] {follow_success}")

    # ---------------------------------------------------------
    # 결과 저장 (Backswing이 없으면 -1 또는 빈 값으로 저장)
    # ---------------------------------------------------------
    csv_rows = []
    
    # Ready
    if 'ready' in results:
        csv_rows.append({'keyframe': 'ready', 'value': int(results['ready'][COLUMN_MAPPING['frame']])})
        visualize_all_keypoints(results['ready'], 'ready', OUTPUT_DIR)
        
    # Backswing (없으면 -1로 기록 -> 점수 산출기에서 체크)
    if 'backswing' in results:
        csv_rows.append({'keyframe': 'backswing', 'value': int(results['backswing'][COLUMN_MAPPING['frame']])})
        visualize_all_keypoints(results['backswing'], 'backswing', OUTPUT_DIR)
    else:
        # 중요! 백스윙을 못 찾았다는 표시
        csv_rows.append({'keyframe': 'backswing', 'value': -1}) 
        
    # Impact
    if 'impact' in results:
        csv_rows.append({'keyframe': 'impact', 'value': int(results['impact'][COLUMN_MAPPING['frame']])})
        visualize_all_keypoints(results['impact'], 'impact', OUTPUT_DIR)

    # Follow
    csv_rows.append({'keyframe': 'followswing', 'value': follow_success})
    
    for key in ['ready', 'backswing', 'impact']:
        if key in results:
            visualize_all_keypoints(results[key], key, OUTPUT_DIR)
            csv_rows.append({'keyframe': key, 'value': int(results[key][COLUMN_MAPPING['frame']])})
            
    csv_rows.append({'keyframe': 'followswing', 'value': follow_success})

    res_df = pd.DataFrame(csv_rows)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    res_df.to_csv(save_path, index=False)
    print(f"모든 처리 완료. 결과 저장됨: {save_path}")

if __name__ == "__main__":
    run_analysis()