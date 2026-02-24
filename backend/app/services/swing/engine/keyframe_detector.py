import pandas as pd
import numpy as np
from typing import Dict, Optional

class KeyframeDetector:
    """키프레임 감지 (Ready, Backswing, Impact)"""
    
    @staticmethod
    def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y) -> float:
        """
        3점으로 이루는 각도 계산
        
        Args:
            a, b, c: 3개의 점 (b가 꼭지점)
            
        Returns:
            각도 (degree)
        """
        a = np.array([a_x, a_y])
        b = np.array([b_x, b_y])
        c = np.array([c_x, c_y])
        
        ba = a - b
        bc = c - b
        
        n_ba = np.linalg.norm(ba)
        n_bc = np.linalg.norm(bc)
        
        if n_ba == 0 or n_bc == 0:
            return 0
        
        cos_angle = np.dot(ba, bc) / (n_ba * n_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))
    
    def detect(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        DataFrame에서 3개 키프레임 감지
        
        Args:
            df: keypoints DataFrame (columns: frame_id, nose_x, nose_y, ...)
            
        Returns:
            {
                'ready': 30,
                'backswing': 48,
                'impact': 60
            }
        """
        if df.empty:
            return None
        
        # 필요한 컬럼 정의
        cols = {
            'nose': ('nose_x', 'nose_y'),
            'shld': ('right_shoulder_x', 'right_shoulder_y'),
            'elb': ('right_elbow_x', 'right_elbow_y'),
            'wri': ('right_wrist_x', 'right_wrist_y'),
            'hand': ('right_pinky_x', 'right_pinky_y')
        }
        
        # 컬럼 존재 여부 확인
        for col_pair in cols.values():
            if col_pair[0] not in df.columns or col_pair[1] not in df.columns:
                print(f"❌ 필요한 컬럼 없음: {col_pair}")
                return None
        
        # 손목 움직임 계산
        df['wrist_move'] = np.sqrt(
            df[cols['wri'][0]].diff()**2 + 
            df[cols['wri'][1]].diff()**2
        ).fillna(1.0)
        
        # 1. 기준점: 손목 최고점 (y 최소값)
        highest_idx = df[cols['wri'][1]].idxmin()
        
        # 2. READY: 최고점 이전, 팔꿈치가 어깨보다 뒤, 정지 상태
        ready_range = df.iloc[:highest_idx].copy()
        
        if ready_range.empty:
            r_idx = 0
        else:
            ready_cands = ready_range[
                ready_range[cols['elb'][0]] < ready_range[cols['shld'][0]]
            ]
            
            if not ready_cands.empty:
                r_idx = ready_cands['wrist_move'].idxmin()
            else:
                r_idx = ready_range['wrist_move'].idxmin()
        
        # 3. IMPACT: 최고점 이후, 손목이 팔꿈치보다 낮아짐
        after_highest_df = df.iloc[highest_idx:].copy()
        
        if after_highest_df.empty:
            i_idx = highest_idx
        else:
            impact_window = after_highest_df[
                after_highest_df[cols['wri'][1]] >= after_highest_df[cols['elb'][1]]
            ].copy()
            
            if not impact_window.empty:
                # 손목-팔꿈치-손가락 각도 계산 (160.5도에 가까울수록)
                impact_window['snap_angle'] = impact_window.apply(
                    lambda r: self.calculate_angle(
                        r[cols['elb'][0]], r[cols['elb'][1]],
                        r[cols['wri'][0]], r[cols['wri'][1]],
                        r[cols['hand'][0]], r[cols['hand'][1]]
                    ), axis=1
                )
                impact_window['angle_diff'] = abs(impact_window['snap_angle'] - 160.5)
                i_idx = impact_window['angle_diff'].idxmin()
            else:
                i_idx = highest_idx
        
        # 4. BACKSWING: Ready와 최고점 사이, 팔꿈치가 가장 높은 순간
        bs_range = df.iloc[r_idx:highest_idx+1].copy()
        
        if bs_range.empty or len(bs_range) < 2:
            b_idx = int((r_idx + highest_idx) / 2) if highest_idx > r_idx else r_idx
        else:
            # 조건: 팔꿈치 < 코 X, 손목 < 코 X, 팔꿈치 < 코 Y (코보다 뒤, 위)
            bs_cands = bs_range[
                (bs_range[cols['elb'][0]] < bs_range[cols['nose'][0]]) & 
                (bs_range[cols['wri'][0]] < bs_range[cols['nose'][0]]) & 
                (bs_range[cols['elb'][1]] < bs_range[cols['nose'][1]])
            ].copy()
            
            if not bs_cands.empty:
                # 팔꿈치가 가장 높은 순간 (y 최소값)
                b_idx = bs_cands[cols['elb'][1]].idxmin()
            else:
                # 차선책: 팔꿈치가 어깨보다 위
                fallback_cands = bs_range[
                    bs_range[cols['elb'][1]] < bs_range[cols['shld'][1]]
                ].copy()
                
                if not fallback_cands.empty:
                    b_idx = fallback_cands[cols['elb'][1]].idxmin()
                else:
                    b_idx = int((r_idx + highest_idx) / 2)
        
        result = {
            'ready': int(r_idx),
            'backswing': int(b_idx),
            'impact': int(i_idx)
        }
        
        print(f"✅ 키프레임 감지 완료: {result}")
        
        return result