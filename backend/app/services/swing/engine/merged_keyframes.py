"""
키프레임 감지: Ready, Backswing, Impact, Follow-through
"""

import pandas as pd
import numpy as np
import cv2
import os
from typing import Dict, Optional


class KeyframeDetector:
    """키프레임 감지 (Ready, Backswing, Impact, Follow-through)"""
    
    def __init__(self):
        # 스켈레톤 연결 정의 (발 포함)
        self.skeleton_connections = [
            # 상체
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # 하체
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            
            # 왼발 (발목-뒤꿈치-발가락 삼각형)
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot_index'),
            ('left_heel', 'left_foot_index'),
            
            # 오른발 (발목-뒤꿈치-발가락 삼각형)
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot_index'),
            ('right_heel', 'right_foot_index'),
        ]
    
    def detect(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        DataFrame에서 키프레임 감지
        
        Args:
            df: keypoints DataFrame (columns: frame_id, nose_x, nose_y, ...)
            
        Returns:
            {
                'ready': 30,
                'backswing': 48,
                'impact': 60,
                'follow_through': 'O' or 'X'
            }
        """
        
        if df.empty:
            print("❌ DataFrame이 비어있습니다!")
            return None
        
        # 좌표 컬럼 단축어
        c = {
            'nose': ['nose_x', 'nose_y'],
            'shld': ['right_shoulder_x', 'right_shoulder_y'],
            'elb': ['right_elbow_x', 'right_elbow_y'],
            'wri': ['right_wrist_x', 'right_wrist_y'],
            'l_wri': ['left_wrist_x', 'left_wrist_y'],
            'l_elb': ['left_elbow_x', 'left_elbow_y']
        }
        
        # 필요한 컬럼 확인
        required_cols = []
        for col_pair in c.values():
            required_cols.extend(col_pair)
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ 필요한 컬럼 없음: {missing}")
            return None
        
        # [STEP 0] 기준점: 최고점 (손목 Y 최소)
        highest_idx = df[c['wri'][1]].idxmin()
        
        # [STEP 1] READY: 왼손 들기 → 팔꿈치 당김
        ready_scope = df.loc[:highest_idx].copy()
        
        # 왼손이 왼팔꿈치보다 위에 있는 구간
        high_hand_filter = ready_scope[
            ready_scope[c['l_wri'][1]] < ready_scope[c['l_elb'][1]]
        ]
        
        target_range = high_hand_filter if not high_hand_filter.empty else ready_scope
        
        if not target_range.empty:
            # 팔꿈치를 가장 뒤로(X 최소) 당긴 지점
            r_idx = int(
                (target_range[c['elb'][0]] - target_range[c['shld'][0]]).idxmin()
            )
        else:
            r_idx = 0

        # [STEP 2] BACKSWING: Ready ~ Highest 사이 (← 먼저!)
        start_bs = min(r_idx, highest_idx)
        bs_scope = df.loc[start_bs:highest_idx].copy()
        
        # 1차: 코보다 뒤 & 위
        cond1 = (
            (bs_scope[c['elb'][0]] < bs_scope[c['nose'][0]]) &
            (bs_scope[c['wri'][0]] < bs_scope[c['nose'][0]]) &
            (bs_scope[c['elb'][1]] < bs_scope[c['nose'][1]])
        )
        cands1 = bs_scope[cond1]
        
        if not cands1.empty:
            b_idx = int(cands1[c['elb'][1]].idxmin())
        else:
            # 2차: 어깨보다 위
            cands2 = bs_scope[bs_scope[c['elb'][1]] < bs_scope[c['shld'][1]]]
            if not cands2.empty:
                b_idx = int(cands2[c['elb'][1]].idxmin())
            else:
                # 3차: 중간값
                b_idx = int((start_bs + highest_idx) / 2)

        # [STEP 3] IMPACT: 백스윙 이후 ~ 최고점 사이에서 팔 가장 펴진 순간
        search_start = b_idx  # 백스윙 프레임
        search_end = int(highest_idx)  # 손목 최고점
        
        impact_scope = df.loc[search_start:search_end].copy()
        
        if not impact_scope.empty:
            # 팔꿈치 각도 계산
            def calc_elbow_angle(row):
                s = np.array([row[c['shld'][0]], row[c['shld'][1]]])
                e = np.array([row[c['elb'][0]], row[c['elb'][1]]])
                w = np.array([row[c['wri'][0]], row[c['wri'][1]]])
                
                v1 = s - e
                v2 = w - e
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            angles = impact_scope.apply(calc_elbow_angle, axis=1)
            i_idx = int(angles.idxmax())  # 팔이 가장 펴진 순간
        else:
            i_idx = int(highest_idx)
        
        # [STEP 4] FOLLOW-THROUGH: Z축 부호 반전
        ft_res = "X"
        
        if 'right_wrist_z' in df.columns:
            init_z = df.loc[i_idx, 'right_wrist_z']
            
            if not pd.isna(init_z) and init_z != 0:
                init_sign = np.sign(init_z)
                
                for z_val in df.loc[i_idx:, 'right_wrist_z']:
                    if pd.isna(z_val):
                        continue
                    
                    if np.sign(z_val) != init_sign and np.sign(z_val) != 0:
                        ft_res = "O"
                        break
        
        result = {
            'ready': int(r_idx),
            'backswing': int(b_idx),
            'impact': int(i_idx),
            'follow_through': ft_res
        }
        
        print(f"✅ 키프레임 감지 완료: {result}")
        
        return result
        
       
    
    def draw_skeleton(self, frame_data, canvas, draw_w, draw_h, off_x, off_y):
        """
        스켈레톤 그리기 (발 포함)
        
        Args:
            frame_data: 프레임 데이터 (Series)
            canvas: 그릴 이미지
            draw_w, draw_h: 그리기 영역 크기
            off_x, off_y: 오프셋
        """
        
        pts_px = {}
        
        # 1. 모든 관절 포인트 찍기
        for col in frame_data.index:
            if col.endswith('_x'):
                base = col[:-2]  # '_x' 제거
                y_col = base + '_y'
                
                # 데이터 유효성 검사
                if (y_col in frame_data.index and
                    not pd.isna(frame_data[col]) and
                    not pd.isna(frame_data[y_col]) and
                    frame_data[col] != 0 and
                    frame_data[y_col] != 0):
                    
                    px = int(off_x + frame_data[col] * draw_w)
                    py = int(off_y + frame_data[y_col] * draw_h)
                    pts_px[base] = (px, py)
                    
                    # 색상 구분: 코(빨강), 그 외(초록)
                    color = (0, 0, 255) if 'nose' in base else (0, 255, 0)
                    cv2.circle(canvas, (px, py), 4, color, -1)
        
        # 2. 선 긋기 (발 포함)
        for s, e in self.skeleton_connections:
            if s in pts_px and e in pts_px:
                cv2.line(canvas, pts_px[s], pts_px[e], (255, 255, 255), 2)
    
    def generate_keyframe_images(
        self,
        df: pd.DataFrame,
        keyframes: Dict,
        output_dir: str,
        base_name: str
    ):
        """
        키프레임 이미지 생성 (검은 배경 + 스켈레톤)
        
        Args:
            df: keypoints DataFrame
            keyframes: {'ready': 30, 'backswing': 48, 'impact': 60}
            output_dir: 출력 폴더
            base_name: 파일명 prefix
        """
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        W, H, padding = 400, 600, 0.15
        draw_w, draw_h = W * (1 - 2 * padding), H * (1 - 2 * padding)
        off_x, off_y = W * padding, H * padding
        
        for label, frame_idx in keyframes.items():
            if label == 'follow_through' or frame_idx is None:
                continue
            
            try:
                row = df.loc[frame_idx]
            except KeyError:
                print(f"⚠️ 프레임 {frame_idx} 없음")
                continue
            
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            self.draw_skeleton(row, canvas, draw_w, draw_h, off_x, off_y)
            
            cv2.putText(
                canvas,
                f"{label.upper()} F:{frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            output_path = os.path.join(output_dir, f"{base_name}_{label}.jpg")
            cv2.imwrite(output_path, canvas)
            print(f"   💾 저장: {output_path}")
