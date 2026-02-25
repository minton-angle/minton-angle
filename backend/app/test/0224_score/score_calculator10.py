import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import json
import math

# ==============================================================================
# [1] 사용자 설정
# ==============================================================================
KEYFRAME_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/minji_b/minji_b_backswing_no_results_1.csv'
ORIGINAL_VIDEO_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/minji_b/minji_b.mp4'
OUTPUT_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/minji_b/minji_b_score_result_10'

MP_MAP = {
    'nose': 0, 'r_sh': 12, 'r_el': 14, 'r_wr': 16, 'r_fin': 20,
    'l_sh': 11, 'l_el': 13, 'l_wr': 15, 'r_hip': 24, 'l_hip': 23,
    'r_ank': 28, 'l_ank': 27
}

class GolfAnalyzer:
    def __init__(self, video_path, keyframe_path, output_dir):
        self.video_path = video_path
        self.keyframe_path = keyframe_path
        self.output_dir = output_dir
        
        # 1. 정지 이미지용
        self.pose_static = mp.solutions.pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.4
        )
        # 2. 연속 영상용
        self.pose_video = mp.solutions.pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.4, 
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(video_path)
        self.keyframes = self._load_keyframes()
        
        self.report = {
            "total_score": 0,
            "details": {
                "Ready": {}, "Rotation": {}, "Backswing": {},
                "Impact": {}, "FollowSwing": {}
            }
        }
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def _load_keyframes(self):
        if not os.path.exists(self.keyframe_path): return {}
        try:
            df = pd.read_csv(self.keyframe_path)
            return df.set_index('keyframe')['value'].to_dict()
        except Exception:
            return {}

    def _get_frame_and_landmarks(self, frame_idx):
        target_f = int(frame_idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = self.cap.read()
        if not ret: return None, None
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_static.process(img_rgb)
        
        landmarks = {}
        if results.pose_landmarks:
            for name, idx in MP_MAP.items():
                landmarks[name] = results.pose_landmarks.landmark[idx]
        return frame, landmarks

    # --- 수학 연산 유틸리티 ---
    def calc_angle(self, p1, p2, p3):
        a, b, c = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)))

    def calculate_ratio(self, target_joint, sh_r, sh_l):
        sh_w = abs(sh_r.x - sh_l.x)
        if sh_w == 0: return 0.0
        sh_y_avg = (sh_r.y + sh_l.y) / 2
        return (sh_y_avg - target_joint.y) / sh_w
    
    def calc_angle_from_horizontal(self, center, target):
        dy = target.y - center.y
        dx = target.x - center.x
        if dx == 0: return 90.0
        angle_rad = math.atan2(abs(dy), abs(dx))
        return math.degrees(angle_rad)

    # --- 시각화 유틸리티 ---
    def draw_text(self, img, text, pos, color=(0, 0, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def draw_line_and_points(self, img, lms, points_chain, color=(0, 0, 255), thickness=3):
        h, w = img.shape[:2]
        coords = []
        for name in points_chain:
            if name in lms:
                coords.append((int(lms[name].x * w), int(lms[name].y * h)))
            else: return []
        for i in range(len(coords) - 1):
            cv2.line(img, coords[i], coords[i+1], color, thickness)
        for x, y in coords:
            cv2.circle(img, (x, y), 5, color, -1)
        return coords

    # --------------------------------------------------------------------------
    # 1. Ready Phase 분석
    # --------------------------------------------------------------------------
    def analyze_ready(self):
        kf = int(self.keyframes.get('ready', -1))
        if kf == -1: return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: return 0

        scores = []
        # (1) 팔 각도
        ang = self.calc_angle(lms['r_wr'], lms['r_el'], lms['l_wr'])
        s_ang = 100 if 18 <= ang <= 70 else max(10, 100 - (int((ang - 70)/10) + 1)*20 if ang > 70 else 100 - (int((18 - ang)/2) + 1)*20)
        self.report['details']['Ready']['Arm_Angle'] = {
            "measured": round(ang, 2), "target": "18 ~ 70", "diff": round(ang - 70 if ang > 70 else (18 - ang if ang < 18 else 0), 2), "score": int(s_ang)
        }
        scores.append(s_ang)

        # (2) 왼쪽 손목 높이
        h_diff = lms['l_wr'].y - lms['l_sh'].y
        s_height = 100 if h_diff < 0 else max(10, 100 - (int(h_diff / 0.05) + 1) * 20)
        self.report['details']['Ready']['Left_Wrist_Height'] = {
            "measured": round(h_diff, 4), "target": "< 0", "diff": round(max(0, h_diff), 4), "score": int(s_height)
        }
        scores.append(s_height)

        # (3) 스탠스 너비
        sh_w = abs(lms['r_sh'].x - lms['l_sh'].x)
        ft_w = abs(lms['r_ank'].x - lms['l_ank'].x)
        s_stance = 100 if ft_w > sh_w else max(10, 100 - (int((sh_w - ft_w) / 0.02) + 1) * 20)
        self.report['details']['Ready']['Stance_Width'] = {
            "measured": round(ft_w, 4), "target": f"> {round(sh_w, 4)}", "diff": round(max(0, sh_w - ft_w), 4), "score": int(s_stance)
        }
        scores.append(s_stance)

        # (4) 손목 높이 비율
        ratio = self.calculate_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        t_min, t_max = -0.5, 2.0
        s_wr_h = 100 if t_min <= ratio <= t_max else max(10, 100 - (int(min(abs(ratio-t_min), abs(ratio-t_max)) / 0.05) + 1) * 20)
        self.report['details']['Ready']['Wrist_Height_Ratio'] = {
            "measured": round(ratio, 2), "target": f"{t_min} ~ {t_max}", "diff": round(min(abs(ratio-t_min), abs(ratio-t_max)) if s_wr_h < 100 else 0, 2), "score": int(s_wr_h)
        }
        scores.append(s_wr_h)

        coords = self.draw_line_and_points(frame, lms, ['r_wr', 'r_el', 'l_wr'], color=(0, 0, 255))
        if coords and len(coords) >= 2:
            el_x, el_y = coords[1]
            self.draw_text(frame, f"{int(ang)} deg", (el_x + 10, el_y), color=(0, 0, 255))

        cv2.imwrite(os.path.join(self.output_dir, "1_Ready.jpg"), frame)
        return sum(scores) / len(scores)

    # --------------------------------------------------------------------------
    # 2. Swing Sequence (이미지 & Rotation & Backswing)
    # --------------------------------------------------------------------------
    def analyze_swing_sequence(self):
        ready_f = int(self.keyframes.get('ready', -1))
        back_f = int(self.keyframes.get('backswing', -1))
        impact_f = int(self.keyframes.get('impact', -1))
        
        # 최소한 Ready, Impact는 있어야 분석 가능
        if ready_f == -1 or impact_f == -1: return 0

        rot_scores, bs_scores = [], []
        
        # ----------------------------------------------------------------------
        # [A] 이미지 추출 로직 (조건 2 반영: 백스윙 불확실 시 균등 분할)
        # ----------------------------------------------------------------------
        seq_frames = []
        is_backswing_valid = (back_f != -1 and ready_f < back_f < impact_f)
        
        if is_backswing_valid:
            # 백스윙이 확실하면: Ready(1) -> 2:1내분점(2) -> Backswing(3) -> 이후 3장
            f1 = ready_f
            f3 = back_f
            f2 = int((ready_f + 2 * back_f) / 3) 
            seq_frames.extend([f1, f2, f3])
            
            # Backswing ~ Impact 구간 3장 추가
            bi_frames = np.linspace(back_f, impact_f, num=4, dtype=int)[1:]
            seq_frames.extend(bi_frames)
        else:
            # 백스윙이 없거나 순서가 이상하면: Ready ~ Impact 사이를 균등하게 6장 추출
            seq_frames = np.linspace(ready_f, impact_f, num=6, dtype=int)
            print("⚠️ 백스윙 정보가 없거나 불안정하여 균등 간격으로 시퀀스 이미지를 생성합니다.")

        # 이미지 저장 및 선 그리기
        for idx, f_num in enumerate(seq_frames):
            frame, lms = self._get_frame_and_landmarks(f_num)
            if frame is not None and lms:
                self.draw_line_and_points(frame, lms, ['r_sh', 'l_sh'], color=(255, 0, 0), thickness=3)
                self.draw_line_and_points(frame, lms, ['r_hip', 'l_hip'], color=(0, 0, 255), thickness=3)
                self.draw_line_and_points(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 255, 255), thickness=3)
                cv2.imwrite(os.path.join(self.output_dir, f"2_Sequence_{idx+1}.jpg"), frame)


        # ----------------------------------------------------------------------
        # [B] Rotation(회전) 판단 로직 (조건 3 반영: 어깨도 스캔 방식 적용)
        # ----------------------------------------------------------------------
        
        # 스캔 범위 설정: Backswing이 있으면 Backswing부터, 없으면 Ready부터 스캔
        scan_start = back_f if is_backswing_valid else ready_f
        scan_end = min(impact_f + 5, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        
        min_hip_diff = 1.0     # 골반 최소 차이 (초기값)
        min_sh_diff = 1.0      # 어깨 최소 차이 (초기값)

        for f in range(scan_start, scan_end + 1):
            frame, lms = self._get_frame_and_landmarks(f)
            if lms:
                # 1. 골반 회전 체크 (X좌표 차이)
                h_diff = abs(lms['r_hip'].x - lms['l_hip'].x)
                if h_diff < min_hip_diff: min_hip_diff = h_diff
                
                # 2. 어깨 회전 체크 (X좌표 차이) - [수정됨: 스캔 방식]
                s_diff = abs(lms['r_sh'].x - lms['l_sh'].x)
                if s_diff < min_sh_diff: min_sh_diff = s_diff

        # --- 골반 점수 ---
        if min_hip_diff <= 0.02: s_hip = 100
        else: s_hip = max(0, 100 - int((min_hip_diff - 0.02) * 800))
        
        self.report['details']['Rotation']['Hip_Max_Rotation'] = {
            "min_diff": round(min_hip_diff, 4), "target": "< 0.02", "score": int(s_hip)
        }
        rot_scores.append(s_hip)

        # --- 어깨 점수 (조건 3: 골반과 동일 로직) ---
        # 어깨는 골반보다 넓으므로 기준을 살짝 완화할 수도 있지만, 요청대로 동일 로직 적용
        # (혹시 너무 점수가 짜다면 0.02를 0.03~0.04로 조정 가능)
        if min_sh_diff <= 0.03: s_sh = 100
        else: s_sh = max(0, 100 - int((min_sh_diff - 0.03) * 500))
        
        self.report['details']['Rotation']['Shoulder_Max_Rotation'] = {
            "min_diff": round(min_sh_diff, 4), "target": "< 0.03", "score": int(s_sh)
        }
        rot_scores.append(s_sh)


        # ----------------------------------------------------------------------
        # [C] Backswing(백스윙) 판단 로직 (조건 1 반영: 0점 연쇄 적용)
        # ----------------------------------------------------------------------
        bs_lms = None
        if is_backswing_valid:
            _, bs_lms = self._get_frame_and_landmarks(back_f)

        if bs_lms:
            # (1) 팔꿈치 리프트 (가장 중요)
            e_ratio = self.calculate_ratio(bs_lms['r_el'], bs_lms['r_sh'], bs_lms['l_sh'])
            t_lift_min, t_lift_max = 1.5, 3.0
            
            if t_lift_min <= e_ratio <= t_lift_max:
                s_lift = 100
            else:
                diff_lift = min(abs(e_ratio - t_lift_min), abs(e_ratio - t_lift_max))
                s_lift = max(0, 100 - int(diff_lift * 400))
                
            self.report['details']['Backswing']['Elbow_Lift'] = {
                "measured": round(e_ratio, 2), "target": f"{t_lift_min} ~ {t_lift_max}", "score": int(s_lift)
            }
            bs_scores.append(s_lift)

            # --- [수정됨] 조건 1: Lift 점수가 0이면 나머지 다 0점 처리 ---
            if s_lift == 0:
                s_wx = 0
                s_bs_ang = 0
                msg_wx = "0 (Dependency: Elbow Lift Failed)"
                msg_ang = "0 (Dependency: Elbow Lift Failed)"
            else:
                # (2) 손목 X축 깊이
                wx_diff = bs_lms['r_wr'].x - bs_lms['nose'].x
                s_wx = 100 if wx_diff < 0 else max(0, 100 - (int(wx_diff / 0.02) + 1) * 20)
                msg_wx = "< 0"

                # (3) L자 각도
                bs_ang = self.calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr'])
                t_ang_min, t_ang_max = 60, 110
                if t_ang_min <= bs_ang <= t_ang_max:
                    s_bs_ang = 100
                else:
                    diff_ang = min(abs(bs_ang - t_ang_min), abs(bs_ang - t_ang_max))
                    s_bs_ang = max(0, 100 - int(diff_ang * 4.0))
                msg_ang = f"{t_ang_min} ~ {t_ang_max}"

            # 결과 리포트 저장
            self.report['details']['Backswing']['Wrist_X_Depth'] = {
                "measured": round(bs_lms['r_wr'].x - bs_lms['nose'].x, 4), "target": msg_wx, "score": int(s_wx)
            }
            bs_scores.append(s_wx)
            
            self.report['details']['Backswing']['L_Shape_Angle'] = {
                "measured": round(self.calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr']), 2), 
                "target": msg_ang, "score": int(s_bs_ang)
            }
            bs_scores.append(s_bs_ang)

        else:
            print("⚠️ 백스윙 프레임이 유효하지 않아 관련 점수를 0점 처리합니다.")

        # 전체 점수 합산 반환
        all_scores = rot_scores + bs_scores
        if all_scores:
            return sum(all_scores) / len(all_scores)
        return 0

    # --------------------------------------------------------------------------
    # 3. Impact Phase 분석
    # --------------------------------------------------------------------------
    def analyze_impact(self):
        kf = int(self.keyframes.get('impact', -1))
        if kf == -1: return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: return 0
        
        scores = []
        ang = self.calc_angle(lms['r_sh'], lms['r_el'], lms['r_wr'])
        s_ang = 100 if 140 <= ang <= 180 else max(10, 100 - int(abs(140 - ang) * 4))
        self.report['details']['Impact']['Arm_Extension_Angle'] = {
            "measured": round(ang, 2), "target": "140 ~ 180", "diff": round(max(0, 140 - ang), 2), "score": int(s_ang)
        }
        scores.append(s_ang)

        w_ratio = self.calculate_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        t_min, t_max = 2.5, 4.5
        s_wr = 100 if t_min <= w_ratio <= t_max else max(10, 100 - (int(min(abs(w_ratio-t_min), abs(w_ratio-t_max)) / 0.5) + 1) * 20)
        self.report['details']['Impact']['Wrist_Height_Ratio'] = {
            "measured": round(w_ratio, 2), "target": f"{t_min} ~ {t_max}", "diff": round(min(abs(w_ratio-t_min), abs(w_ratio-t_max)) if s_wr < 100 else 0, 2), "score": int(s_wr)
        }
        scores.append(s_wr)
        
        coords = self.draw_line_and_points(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 0, 255))
        if coords and len(coords) >= 2:
            el_x, el_y = coords[1]
            self.draw_text(frame, f"{int(ang)} deg", (el_x, el_y - 20), (0, 0, 255))
        
        cv2.imwrite(os.path.join(self.output_dir, "3_Impact.jpg"), frame)
        return sum(scores) / len(scores)

    # --------------------------------------------------------------------------
    # 4. Follow Swing 분석
    # --------------------------------------------------------------------------
    def analyze_follow(self):
        start_f = int(self.keyframes.get('impact', -1))
        if start_f == -1: return 0
        
        end_f = min(start_f + 40, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        w, h = int(self.cap.get(3)), int(self.cap.get(4))
        fps = self.cap.get(5)
        
        out = cv2.VideoWriter(os.path.join(self.output_dir, "4_FollowSwing.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        has_reached_angle_30 = False
        has_crossed_x = False

        for i in range(start_f, end_f + 1):
            ret, frame = self.cap.read()
            if not ret: break
            
            res = self.pose_video.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = {name: res.pose_landmarks.landmark[idx] for name, idx in MP_MAP.items()}
                self.draw_line_and_points(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 0, 255), thickness=3)
                
                angle_horiz = self.calc_angle_from_horizontal(lms['r_sh'], lms['r_wr'])
                
                if angle_horiz > 30: has_reached_angle_30 = True
                if lms['r_wr'].x <= lms['r_el'].x: has_crossed_x = True

                self.draw_text(frame, f"Ang: {int(angle_horiz)}", (50, 50), color=(0, 255, 0) if angle_horiz > 30 else (0, 0, 255))

            out.write(frame)
        out.release()
        
        score = 0
        msg = ""

        if not has_reached_angle_30:
            score = 0
            msg = "Follow Swing Too Low (Angle < 30)"
        elif not has_crossed_x:
            score = 50
            msg = "Angle > 30 but Wrist did not cross elbow"
        else:
            score = 100
            msg = "Perfect Follow Swing (Angle > 30 & Crossed)"

        self.report['details']['FollowSwing']['Performance'] = {
            "score": score, "success": (score == 100), "message": msg,
            "has_reached_30": has_reached_angle_30, "has_crossed_x": has_crossed_x
        }
        
        return score

    def run(self):
        if not self.cap.isOpened(): return
        print("--- [1] Ready Phase ---")
        s1 = self.analyze_ready()
        print("--- [2] Swing Sequence ---")
        s2 = self.analyze_swing_sequence()
        print("--- [3] Impact Phase ---")
        s3 = self.analyze_impact()
        print("--- [4] Follow Swing ---")
        s4 = self.analyze_follow()

        s1 = s1 if s1 is not None else 0
        s2 = s2 if s2 is not None else 0
        s3 = s3 if s3 is not None else 0
        s4 = s4 if s4 is not None else 0
      
        self.report['total_score'] = round((s1 + s2 + s3 + s4) / 4, 1)
        
        with open(os.path.join(self.output_dir, "final_analysis_report.json"), 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ 최종 분석 완료! 결과물 저장 경로: {self.output_dir}")
        self.cap.release()
        self.pose_static.close()
        self.pose_video.close()

if __name__ == "__main__":
    analyzer = GolfAnalyzer(ORIGINAL_VIDEO_PATH, KEYFRAME_CSV_PATH, OUTPUT_DIR)
    analyzer.run()