import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import json

# ==============================================================================
# [1] 사용자 설정
# ==============================================================================
KEYFRAME_CSV_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/wooil/wooil_backswing_no_results_1.csv'
ORIGINAL_VIDEO_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/wooil/wooil.mp4'
OUTPUT_DIR = '/Users/minji/Documents/minton-angle/backend/data/standard/0224_resources/wooil/wooil_score_result_3'

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
        
        # 1. 정지 이미지용 (Ready, Sequence, Impact 추출 시 사용)
        self.pose_static = mp.solutions.pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.4
        )
        # 2. 연속 영상용 (FollowSwing 영상 생성 시 사용)
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
    # 1. Ready Phase 분석 (감점 20점 단위로 강화)
    # --------------------------------------------------------------------------
    def analyze_ready(self):
        kf = int(self.keyframes.get('ready', -1))
        if kf == -1: return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: return 0

        scores = []
        # (1) 팔 각도: 범위를 벗어나면 기존 10점 감점 -> 20점씩 감점
        ang = self.calc_angle(lms['r_wr'], lms['r_el'], lms['l_wr'])
        s_ang = 100 if 18 <= ang <= 70 else max(10, 100 - (int((ang - 70)/10) + 1)*20 if ang > 70 else 100 - (int((18 - ang)/2) + 1)*20)
        self.report['details']['Ready']['Arm_Angle'] = {
            "measured": round(ang, 2), "target": "18 ~ 70", "diff": round(ang - 70 if ang > 70 else (18 - ang if ang < 18 else 0), 2), "score": int(s_ang)
        }
        scores.append(s_ang)

        # (2) 왼쪽 손목 높이: 기준 이탈 시 기존 10점 감점 -> 20점씩 감점
        h_diff = lms['l_wr'].y - lms['l_sh'].y
        s_height = 100 if h_diff < 0 else max(10, 100 - (int(h_diff / 0.05) + 1) * 20)
        self.report['details']['Ready']['Left_Wrist_Height'] = {
            "measured": round(h_diff, 4), "target": "< 0", "diff": round(max(0, h_diff), 4), "score": int(s_height)
        }
        scores.append(s_height)

        # (3) 스탠스 너비: 기준 이탈 시 기존 10점 감점 -> 20점씩 감점
        sh_w = abs(lms['r_sh'].x - lms['l_sh'].x)
        ft_w = abs(lms['r_ank'].x - lms['l_ank'].x)
        s_stance = 100 if ft_w > sh_w else max(10, 100 - (int((sh_w - ft_w) / 0.02) + 1) * 20)
        self.report['details']['Ready']['Stance_Width'] = {
            "measured": round(ft_w, 4), "target": f"> {round(sh_w, 4)}", "diff": round(max(0, sh_w - ft_w), 4), "score": int(s_stance)
        }
        scores.append(s_stance)

        # (4) 손목 높이 비율: 기준 이탈 시 기존 10점 감점 -> 20점씩 감점
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
    # 2. Swing Sequence 분석 (감점 폭 2배로 강화)
    # --------------------------------------------------------------------------
    def analyze_swing_sequence(self):
        ready_f = int(self.keyframes.get('ready', -1))
        back_f = int(self.keyframes.get('backswing', -1))
        impact_f = int(self.keyframes.get('impact', -1))
        
        if ready_f == -1 or impact_f == -1: return 0

        _, rdy_lms = self._get_frame_and_landmarks(ready_f)
        _, imp_lms = self._get_frame_and_landmarks(impact_f)
        
        rot_scores, bs_scores = [], []
        
        # --- 1. Rotation(회전) 판단 로직 ---
        if rdy_lms and imp_lms:
            # 골반 회전 (감점 계수 500 -> 1000으로 2배 강화)
            hip_x_diff = abs(imp_lms['r_hip'].x - imp_lms['l_hip'].x)
            s_hip = 100 if hip_x_diff <= 0.03 else max(10, 100 - int((hip_x_diff - 0.03) * 1000))
            self.report['details']['Rotation']['Hip_Frontal_Alignment'] = {
                "measured_x_diff": round(hip_x_diff, 4),
                "target": "< 0.03 (Facing Front)",
                "score": int(s_hip)
            }
            rot_scores.append(s_hip)

            # 어깨 회전 (감점 계수 150 -> 300으로 2배 강화)
            init_sh_w = abs(rdy_lms['r_sh'].x - rdy_lms['l_sh'].x)
            curr_sh_w = abs(imp_lms['r_sh'].x - imp_lms['l_sh'].x)
            sh_x_diff = abs(imp_lms['r_sh'].x - imp_lms['l_sh'].x)
            w_ratio = curr_sh_w / init_sh_w if init_sh_w != 0 else 1.0
            
            s_sh = 100 if (sh_x_diff <= 0.03 or 0.4 <= w_ratio <= 0.7) else max(10, 100 - int(min(abs(w_ratio - 0.4), abs(w_ratio - 0.7)) * 300))
            self.report['details']['Rotation']['Shoulder_Frontal_Alignment'] = {
                "measured_ratio": round(w_ratio, 2),
                "measured_x_diff": round(sh_x_diff, 4),
                "score": int(s_sh)
            }
            rot_scores.append(s_sh)

        # --- 2. Backswing(백스윙) 판단 로직 ---
        bs_lms = None
        if back_f != -1:
            _, bs_lms = self._get_frame_and_landmarks(back_f)

        if bs_lms:
            # (1) 손목 X축 깊이: 기존 10점 단위 -> 20점 단위 감점
            wx_diff = bs_lms['r_wr'].x - bs_lms['nose'].x
            s_wx = 100 if wx_diff < 0 else max(10, 100 - (int(wx_diff / 0.02) + 1) * 20)
            self.report['details']['Backswing']['Wrist_X_Depth'] = {"measured": round(wx_diff, 4), "target": "< 0", "score": int(s_wx)}
            bs_scores.append(s_wx)

            # (2) 팔꿈치 리프트 비율: 감점 계수 200 -> 400으로 2배 강화
            e_ratio = self.calculate_ratio(bs_lms['r_el'], bs_lms['r_sh'], bs_lms['l_sh'])
            t_lift_min, t_lift_max = 1.5, 3.0
            
            if t_lift_min <= e_ratio <= t_lift_max:
                s_lift = 100
            else:
                diff_lift = min(abs(e_ratio - t_lift_min), abs(e_ratio - t_lift_max))
                s_lift = max(0, 100 - int(diff_lift * 400))
                
            self.report['details']['Backswing']['Elbow_Lift'] = {
                "measured": round(e_ratio, 2), 
                "target": f"{t_lift_min} ~ {t_lift_max}", 
                "score": int(s_lift)
            }
            bs_scores.append(s_lift)

            # (3) 팔꿈치 L자 각도: 감점 계수 2.0 -> 4.0으로 2배 강화 (1도 벗어날 때마다 4점 감점)
            bs_ang = self.calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr'])
            t_ang_min, t_ang_max = 60, 110
            
            if t_ang_min <= bs_ang <= t_ang_max:
                s_bs_ang = 100
            else:
                diff_ang = min(abs(bs_ang - t_ang_min), abs(bs_ang - t_ang_max))
                s_bs_ang = max(0, 100 - int(diff_ang * 4.0))
                
            self.report['details']['Backswing']['L_Shape_Angle'] = {
                "measured": round(bs_ang, 2), 
                "target": f"{t_ang_min} ~ {t_ang_max}", 
                "score": int(s_bs_ang)
            }
            bs_scores.append(s_bs_ang)
        else:
            print("⚠️ 백스윙 프레임이 없어 관련 지표를 0점으로 처리합니다.")

        # --- 3. 시퀀스 이미지 저장 ---
        if back_f != -1:
            seq_2_f = int(ready_f + (back_f - ready_f) * 2 / 3)
            if seq_2_f <= ready_f and back_f > ready_f: seq_2_f = ready_f + 1
            elif seq_2_f >= back_f and back_f > ready_f + 1: seq_2_f = back_f - 1

            bi_step = max(1, (impact_f - back_f) // 3)
            frames_map = {
                "Seq_1_Ready": ready_f, "Seq_2_Takeaway": seq_2_f, "Seq_3_Backswing": back_f,
                "Seq_4_Downswing_1": back_f + bi_step, "Seq_5_Downswing_2": back_f + (bi_step * 2), "Seq_6_Impact": impact_f
            }
        else:
            step = max(1, (impact_f - ready_f) // 5)
            frames_map = {
                "Seq_1_Ready": ready_f, 
                "Seq_2_Takeaway": ready_f + step, 
                "Seq_3_Missing_Backswing": ready_f + step * 2,
                "Seq_4_Downswing_1": ready_f + step * 3, 
                "Seq_5_Downswing_2": ready_f + step * 4, 
                "Seq_6_Impact": impact_f
            }

        for name, f_idx in frames_map.items():
            frame, lms = self._get_frame_and_landmarks(f_idx)
            if frame is not None and lms:
                self.draw_line_and_points(frame, lms, ['r_wr', 'r_el', 'r_sh'], color=(0, 255, 0)) 
                self.draw_line_and_points(frame, lms, ['l_hip', 'r_hip'], color=(0, 0, 255)) 
                self.draw_line_and_points(frame, lms, ['l_sh', 'r_sh'], color=(255, 0, 0)) 
            if frame is not None:
                cv2.imwrite(os.path.join(self.output_dir, f"{name}.jpg"), frame)

        avg_rot = sum(rot_scores) / len(rot_scores) if rot_scores else 0
        avg_bs = sum(bs_scores) / len(bs_scores) if bs_scores else 0
        return (avg_rot + avg_bs) / 2

    # --------------------------------------------------------------------------
    # 3. Impact Phase 분석 (감점 2배 강화)
    # --------------------------------------------------------------------------
    def analyze_impact(self):
        kf = int(self.keyframes.get('impact', -1))
        if kf == -1: return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: return 0
        
        scores = []
        # 팔 펴짐 각도: 1도 이탈 시 2점 감점 -> 4점 감점으로 2배 엄격하게
        ang = self.calc_angle(lms['r_sh'], lms['r_el'], lms['r_wr'])
        s_ang = 100 if 140 <= ang <= 180 else max(10, 100 - int(abs(140 - ang) * 4))
        self.report['details']['Impact']['Arm_Extension_Angle'] = {
            "measured": round(ang, 2), "target": "140 ~ 180", "diff": round(max(0, 140 - ang), 2), "score": int(s_ang)
        }
        scores.append(s_ang)

        # 손목 높이 비율: 기존 10점 단위 -> 20점 단위 감점
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
    # 4. Follow Swing 분석 (영상 및 점수) - [수정됨]
    # --------------------------------------------------------------------------
    def analyze_follow(self):
        start_f = int(self.keyframes.get('impact', -1))
        if start_f == -1: return 0
        
        # 임팩트 이후 40프레임 또는 영상 끝까지 확인
        end_f = min(start_f + 40, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        w, h = int(self.cap.get(3)), int(self.cap.get(4))
        fps = self.cap.get(5)
        
        # 영상 저장을 위한 설정
        out = cv2.VideoWriter(os.path.join(self.output_dir, "4_FollowSwing.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        # [상태 플래그 초기화]
        # 1. Y축: 손목이 팔꿈치보다 아래로 떨어졌는가? (스윙 궤적 확인)
        has_dropped_y = False
        # 2. X축: 손목이 팔꿈치 안쪽으로(또는 일직선상에) 확실히 들어왔는가? (Strict Check)
        has_crossed_x = False

        for i in range(start_f, end_f + 1):
            ret, frame = self.cap.read()
            if not ret: break
            
            res = self.pose_video.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = {name: res.pose_landmarks.landmark[idx] for name, idx in MP_MAP.items()}
                self.draw_line_and_points(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 0, 255), thickness=3)
                
                # --- [판단 로직 변경] 매 프레임마다 확인 ---
                
                # 조건 1: 높이(Y) 체크 - 손목이 팔꿈치 아래로 내려가는지
                if lms['r_wr'].y > lms['r_el'].y: 
                    has_dropped_y = True
                
                # 조건 2: 교차(X) 체크 - 허용오차 없이 겹치거나 넘어서는 순간 포착
                # (일반적으로 백핸드 팔로우스윙 시 손목이 팔꿈치보다 왼쪽(화면상) 혹은 같은 위치까지 와야 함)
                if lms['r_wr'].x <= lms['r_el'].x:
                    has_crossed_x = True

            out.write(frame)
        out.release()
        
        # --- [점수 산출 로직] 0, 50, 100점 체계 ---
        score = 0
        msg = ""

        if not has_dropped_y:
            # 1단계 실패: 손목이 내려오지도 않음
            score = 0
            msg = "Follow Swing Incomplete: Wrist did not drop below elbow"
        elif not has_crossed_x:
            # 1단계 성공, 2단계 실패: 내려는 왔으나 팔꿈치 라인을 넘지 못함 (어정쩡한 스윙)
            score = 50
            msg = "Wrist dropped but did not fully cross the elbow line (Strict Check Fail)"
        else:
            # 1, 2단계 모두 성공: 확실하게 교차함
            score = 100
            msg = "Perfect Follow Swing: Wrist crossed the elbow line"

        self.report['details']['FollowSwing']['Performance'] = {
            "score": score, 
            "success": (score == 100), 
            "message": msg,
            "has_dropped_y": has_dropped_y,
            "has_crossed_x": has_crossed_x
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