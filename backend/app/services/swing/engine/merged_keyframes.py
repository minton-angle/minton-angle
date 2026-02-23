import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 팀원 알고리즘: 키프레임 감지

COLUMN_MAPPING = {
    'frame': 'frame_id',
    'nose_x': 'nose_x', 'nose_y': 'nose_y',
    'r_sh_x': 'right_shoulder_x', 'r_sh_y': 'right_shoulder_y',
    'r_el_x': 'right_elbow_x', 'r_el_y': 'right_elbow_y',
    'r_wr_x': 'right_wrist_x', 'r_wr_y': 'right_wrist_y',
    'r_wr_z': 'right_wrist_z',
    'l_sh_x': 'left_shoulder_x', 'l_sh_y': 'left_shoulder_y',
    'l_el_x': 'left_elbow_x', 'l_el_y': 'left_elbow_y',
    'l_wr_x': 'left_wrist_x', 'l_wr_y': 'left_wrist_y',
    'r_hip_x': 'right_hip_x', 'r_hip_y': 'right_hip_y',
    'l_hip_x': 'left_hip_x', 'l_hip_y': 'left_hip_y',
    'r_knee_x': 'right_knee_x', 'r_knee_y': 'right_knee_y',
    'l_knee_x': 'left_knee_x', 'l_knee_y': 'left_knee_y',
    'r_ank_x': 'right_ankle_x', 'r_ank_y': 'right_ankle_y',
    'l_ank_x': 'left_ankle_x', 'l_ank_y': 'left_ankle_y',
    'r_heel_x': 'right_heel_x', 'r_heel_y': 'right_heel_y',
    'r_foot_idx_x': 'right_foot_index_x', 'r_foot_idx_y': 'right_foot_index_y',
    'l_heel_x': 'left_heel_x', 'l_heel_y': 'left_heel_y',
    'l_foot_idx_x': 'left_foot_index_x', 'l_foot_idx_y': 'left_foot_index_y',
    'r_fin_x': 'right_index_x', 'r_fin_y': 'right_index_y'
}

SKELETON_CONNECTIONS = [
    ('nose', 'r_sh'), ('r_sh', 'r_el'), ('r_el', 'r_wr'), ('r_wr', 'r_fin'),
    ('nose', 'l_sh'), ('l_sh', 'l_el'), ('l_el', 'l_wr'),
    ('r_sh', 'l_sh'), ('r_sh', 'r_hip'), ('l_sh', 'l_hip'), ('r_hip', 'l_hip'),
    ('r_hip', 'r_knee'), ('r_knee', 'r_ank'),
    ('l_hip', 'l_knee'), ('l_knee', 'l_ank'),
    ('r_ank', 'r_heel'), ('r_heel', 'r_foot_idx'), ('r_ank', 'r_foot_idx'),
    ('l_ank', 'l_heel'), ('l_heel', 'l_foot_idx'), ('l_ank', 'l_foot_idx')
]


def get_val(df, key):
    """매핑된 컬럼 데이터 가져오기"""
    col_name = COLUMN_MAPPING.get(key)
    if col_name in df.columns:
        return df[col_name]
    return None


def detect_keyframes_from_df(df: pd.DataFrame) -> dict:
    """
    DataFrame에서 키프레임 감지 (팀원 알고리즘)
    
    Returns:
        {
            'ready': int,
            'backswing': int,
            'impact': int,
            'followswing': 'O' or 'X'
        }
    """
    
    if df.empty:
        print("❌ DataFrame이 비어있습니다!")
        return None
    
    print(f"CSV 로드 완료: {len(df)} frames")
    
    results = {}
    
    sh_x = get_val(df, 'r_sh_x')
    el_x = get_val(df, 'r_el_x')
    wr_y = get_val(df, 'r_wr_y')
    wr_z = get_val(df, 'r_wr_z')
    
    # [1] READY
    relative_dist = sh_x - el_x 
    ready_idx = relative_dist.idxmax()
    
    results['ready'] = df.loc[ready_idx]
    print(f"[Ready] Frame {int(results['ready'][COLUMN_MAPPING['frame']])} (Rel. Dist: {relative_dist[ready_idx]:.2f})")

    # [2] 기준점 (Highest Point)
    highest_idx = wr_y.idxmin()
    
    # [3] BACKSWING
    start_search = ready_idx if ready_idx < highest_idx else 0
    bs_range = df.loc[start_search:highest_idx].copy()
    
    nose_x = bs_range[COLUMN_MAPPING['nose_x']]
    el_x_bs = bs_range[COLUMN_MAPPING['r_el_x']]
    wr_x = bs_range[COLUMN_MAPPING['r_wr_x']]
    el_y = bs_range[COLUMN_MAPPING['r_el_y']]
    sh_y = bs_range[COLUMN_MAPPING['r_sh_y']]

    c1 = el_x_bs < nose_x
    c2 = wr_x < nose_x
    c3 = el_y < sh_y

    bs_cands = bs_range[c1 & c2 & c3]
    
    backswing_frame_val = None

    if not bs_cands.empty:
        backswing_idx = bs_cands[COLUMN_MAPPING['r_el_y']].idxmin()
        backswing_frame_val = int(df.loc[backswing_idx][COLUMN_MAPPING['frame']])
        results['backswing'] = df.loc[backswing_idx]
        print(f"[Backswing] Frame {backswing_frame_val}")
    else:
        print("[Backswing] ⚠️ 감지되지 않음 (조건 미충족)")
        backswing_idx = int((start_search + highest_idx) / 2)
        results['backswing'] = df.loc[backswing_idx]
        backswing_frame_val = int(df.loc[backswing_idx][COLUMN_MAPPING['frame']])
        print(f"[Backswing] Frame {backswing_frame_val} (fallback: 중간값)")

    # [4] IMPACT
    facing_left = results['ready'][COLUMN_MAPPING['nose_x']] < results['ready'][COLUMN_MAPPING['r_sh_x']]

    search_start = max(0, highest_idx - 10)
    search_end = min(len(df) - 1, highest_idx + 20)
    impact_search_range = df.loc[search_start:search_end].copy()

    best_impact_score = -1
    impact_idx = highest_idx

    for idx, row in impact_search_range.iterrows():
        rx, ry = row[COLUMN_MAPPING['r_sh_x']], row[COLUMN_MAPPING['r_sh_y']]
        ex, ey = row[COLUMN_MAPPING['r_el_x']], row[COLUMN_MAPPING['r_el_y']]
        wx, wy = row[COLUMN_MAPPING['r_wr_x']], row[COLUMN_MAPPING['r_wr_y']]

        if facing_left:
            is_in_front = wx < rx
        else:
            is_in_front = wx > rx
        
        if not is_in_front:
            continue

        is_height_hierarchy_ok = (ry > ey) and (ey > wy)
        
        if not is_height_hierarchy_ok:
            continue

        distance = np.sqrt((rx - wx)**2 + (ry - wy)**2)
        height_score = 1.0 - wy 
        
        ba = np.array([rx - ex, ry - ey])
        bc = np.array([wx - ex, wy - ey])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        angle_score = angle / 180.0

        total_score = (distance * 0.4) + (height_score * 0.4) + (angle_score * 0.2)

        if total_score > best_impact_score:
            best_impact_score = total_score
            impact_idx = idx

    results['impact'] = df.loc[impact_idx]
    print(f"[Impact] Frame {int(results['impact'][COLUMN_MAPPING['frame']])} (Score: {best_impact_score:.4f})")

    # [5] FOLLOW-THROUGH
    follow_success = "X"
    follow_range = df.loc[impact_idx:]
    
    if wr_z is not None and not wr_z.isna().all():
        try:
            initial_z = wr_z.loc[impact_idx]
            initial_sign = np.sign(initial_z)
            for idx in follow_range.index:
                curr_z = wr_z.loc[idx]
                if pd.isna(curr_z): 
                    continue
                if np.sign(curr_z) != initial_sign and initial_sign != 0:
                    follow_success = "O"
                    break
        except Exception: 
            pass

    if follow_success == "X":
        fs_check = follow_range[
            follow_range[COLUMN_MAPPING['r_wr_x']] < follow_range[COLUMN_MAPPING['r_sh_x']]
        ]
        if not fs_check.empty:
            follow_success = "O"

    print(f"[Follow-through] {follow_success}")

    return {
        'ready': int(results['ready'][COLUMN_MAPPING['frame']]),
        'backswing': int(results['backswing'][COLUMN_MAPPING['frame']]),
        'impact': int(results['impact'][COLUMN_MAPPING['frame']]),
        'followswing': follow_success
    }