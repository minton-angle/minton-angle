import pandas as pd
import numpy as np
import cv2
import os
from typing import Dict, Optional, Tuple
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class KeyframeDetector:
    """키프레임 감지 (Ready, Backswing, Impact, Follow-through)"""
    
    def __init__(self, target_frames: int = 60):
        """
        Args:
            target_frames: 정규화 목표 프레임 수 (기본 60)
        """
        # 정규화 목표 프레임 수
        self.TARGET_FRAMES = target_frames
        
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
    
    # ============================================
    # 전처리/정규화 메서드들
    # ============================================
    
    def _detect_swing_region(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        스윙 구간 자동 감지 (손목 움직임 기반)
        
        Returns:
            (start_idx, end_idx): 스윙 시작/끝 프레임 인덱스
        """
        wrist_y_col = 'right_wrist_y'
        
        # 컬럼 없으면 전체 반환
        if wrist_y_col not in df.columns:
            print("⚠️ right_wrist_y 컬럼 없음, 전체 구간 사용")
            return 0, len(df) - 1
        
        wrist_y = df[wrist_y_col].values
        
        # 스무딩 (노이즈 제거)
        wrist_y_smooth = gaussian_filter1d(wrist_y, sigma=2)
        
        # 손목 최고점 (Y 최소 = 화면상 가장 위)
        peak_idx = int(np.argmin(wrist_y_smooth))
        
        # 속도 계산 (1차 미분)
        velocity = np.gradient(wrist_y_smooth)
        
        # ============================================
        # 스윙 시작점 찾기: peak 이전에서 손목이 올라가기 시작하는 순간
        # ============================================
        start_idx = 0
        for i in range(peak_idx - 1, 0, -1):
            # 속도가 양수→음수로 바뀌는 지점 (손목이 내려가다가 올라가기 시작)
            # velocity < 0 = 손목이 올라감 (Y 감소)
            if velocity[i] < -0.002 and velocity[i-1] >= 0:
                start_idx = max(0, i - 5)  # 약간 여유
                break
        
        # ============================================
        # 스윙 끝점 찾기: peak 이후에서 손목이 충분히 내려온 지점
        # ============================================
        end_idx = len(df) - 1
        peak_y = wrist_y_smooth[peak_idx]
        start_y = wrist_y_smooth[start_idx]
        
        # 임계값: 시작점과 정점 사이의 70% 지점
        threshold = peak_y + (start_y - peak_y) * 0.7
        
        for i in range(peak_idx, len(wrist_y_smooth)):
            if wrist_y_smooth[i] > threshold:
                end_idx = min(len(df) - 1, i + 5)  # 약간 여유
                break
        
        # 최소 프레임 수 보장 (너무 짧으면 확장)
        min_frames = 20
        if end_idx - start_idx < min_frames:
            start_idx = max(0, peak_idx - min_frames // 2)
            end_idx = min(len(df) - 1, peak_idx + min_frames // 2)
        
        print(f"📍 스윙 구간 감지: {start_idx} ~ {end_idx} (총 {end_idx - start_idx + 1} 프레임)")
        
        return start_idx, end_idx
    
    def _resample_keypoints(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[pd.DataFrame, float, int]:
        """
        스윙 구간을 고정 프레임 수로 리샘플링
        
        Args:
            df: 원본 keypoints DataFrame
            start_idx: 스윙 시작 인덱스
            end_idx: 스윙 끝 인덱스
            
        Returns:
            (리샘플링된 DataFrame, scale 비율, 시작 인덱스)
        """
        # 스윙 구간만 추출
        swing_df = df.loc[start_idx:end_idx].copy().reset_index(drop=True)
        original_frames = len(swing_df)
        
        # 이미 목표 프레임 수면 그대로 반환
        if original_frames == self.TARGET_FRAMES:
            print(f"✅ 프레임 수 동일: {original_frames}")
            return swing_df, 1.0, start_idx
        
        # 프레임 수 너무 적으면 경고
        if original_frames < 10:
            print(f"⚠️ 프레임 수 부족: {original_frames}, 리샘플링 스킵")
            return swing_df, 1.0, start_idx
        
        # ============================================
        # 각 컬럼별로 선형 보간
        # ============================================
        result_data = {}
        original_indices = np.arange(original_frames)
        target_indices = np.linspace(0, original_frames - 1, self.TARGET_FRAMES)
        
        for col in swing_df.columns:
            # 숫자 타입 컬럼만 보간
            if swing_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    # 선형 보간 함수 생성
                    f = interp1d(
                        original_indices, 
                        swing_df[col].values, 
                        kind='linear', 
                        fill_value='extrapolate'
                    )
                    result_data[col] = f(target_indices)
                except Exception as e:
                    print(f"⚠️ 보간 실패 ({col}): {e}")
                    # 실패 시 가장 가까운 값 사용
                    nearest = np.round(target_indices).astype(int)
                    nearest = np.clip(nearest, 0, original_frames - 1)
                    result_data[col] = swing_df[col].iloc[nearest].values
            else:
                # 비숫자 컬럼: 가장 가까운 값 사용
                nearest = np.round(target_indices).astype(int)
                nearest = np.clip(nearest, 0, original_frames - 1)
                result_data[col] = swing_df[col].iloc[nearest].values
        
        result_df = pd.DataFrame(result_data)
        result_df.index = range(self.TARGET_FRAMES)
        
        # scale 계산 (원본 인덱스 복원용)
        scale = original_frames / self.TARGET_FRAMES
        
        print(f"✅ 리샘플링 완료: {original_frames} → {self.TARGET_FRAMES} 프레임 (scale: {scale:.2f})")
        
        return result_df, scale, start_idx
    
    # ============================================
    # 기존 키프레임 감지 알고리즘 (그대로 유지)
    # ============================================
    
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
        search_end = min(int(highest_idx) + 3, len(df) - 1)  # +3 추가

        # 기본값 먼저 할당!
        i_idx = int(highest_idx)
        
        if search_start < search_end:
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
                if not angles.empty:
                    i_idx = int(angles.idxmax())  # 팔이 가장 펴진 순간
        
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
    
    # ============================================
    # 정규화 + 감지 통합 메서드 (새로 추가)
    # ============================================
    
    def detect_with_normalization(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        정규화 적용 후 키프레임 감지
        
        1. 스윙 구간 자동 감지
        2. 고정 프레임 수(60)로 리샘플링
        3. 기존 알고리즘으로 키프레임 감지
        4. 원본 프레임 인덱스로 변환
        
        Args:
            df: 원본 keypoints DataFrame
            
        Returns:
            원본 프레임 인덱스 기준 키프레임 dict
        """
        
        if df.empty:
            print("❌ DataFrame이 비어있습니다!")
            return None
        
        print("=" * 50)
        print("🔄 정규화 키프레임 감지 시작")
        print("=" * 50)
        
        # ============================================
        # 1. 스윙 구간 감지
        # ============================================
        start_idx, end_idx = self._detect_swing_region(df)
        
        # ============================================
        # 2. 고정 프레임 수로 리샘플링
        # ============================================
        normalized_df, scale, swing_start = self._resample_keypoints(df, start_idx, end_idx)
        
        # ============================================
        # 3. 기존 알고리즘으로 키프레임 감지
        # ============================================
        print("\n📊 정규화된 데이터에서 키프레임 감지...")
        norm_keyframes = self.detect(normalized_df)
        
        if norm_keyframes is None:
            print("❌ 키프레임 감지 실패")
            return None
        
        # ============================================
        # 4. 원본 프레임 인덱스로 변환
        # ============================================
        result = {
            'ready': int(swing_start + norm_keyframes['ready'] * scale),
            'backswing': int(swing_start + norm_keyframes['backswing'] * scale),
            'impact': int(swing_start + norm_keyframes['impact'] * scale),
            'follow_through': norm_keyframes['follow_through']
        }
        
        # 범위 체크 (원본 DataFrame 범위 내로)
        max_idx = len(df) - 1
        result['ready'] = max(0, min(result['ready'], max_idx))
        result['backswing'] = max(0, min(result['backswing'], max_idx))
        result['impact'] = max(0, min(result['impact'], max_idx))
        
        print("\n" + "=" * 50)
        print(f"📊 정규화 키프레임 (60프레임 기준): {norm_keyframes}")
        print(f"✅ 원본 키프레임 (변환 후): {result}")
        print("=" * 50)
        
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