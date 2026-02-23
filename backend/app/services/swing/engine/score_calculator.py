import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import json

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
        
        self.pose_static = mp.solutions.pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.4
        )
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
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)

    def _load_keyframes(self):
        if not os.path.exists(self.keyframe_path): 
            return {}
        try:
            df = pd.read_csv(self.keyframe_path)
            return df.set_index('keyframe')['value'].to_dict()
        except Exception:
            return {}

    def _get_frame_and_landmarks(self, frame_idx):
        target_f = int(frame_idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = self.cap.read()
        if not ret: 
            return None, None
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_static.process(img_rgb)
        
        landmarks = {}
        if results.pose_landmarks:
            for name, idx in MP_MAP.items():
                landmarks[name] = results.pose_landmarks.landmark[idx]
        return frame, landmarks

    def calc_angle(self, p1, p2, p3):
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: 
            return 0.0
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def calculate_ratio(self, target_joint, sh_r, sh_l):
        sh_w = abs(sh_r.x - sh_l.x)
        if sh_w == 0: 
            return 0.0
        sh_y_avg = (sh_r.y + sh_l.y) / 2
        return (sh_y_avg - target_joint.y) / sh_w

    def draw_text(self, img, text, pos, color=(0, 0, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2, cv2.LINE_AA)

    def draw_line_and_points(self, img, lms, points_chain, 
                             color=(0, 0, 255), thickness=3):
        h, w = img.shape[:2]
        coords = []
        for name in points_chain:
            if name in lms:
                coords.append((int(lms[name].x * w), int(lms[name].y * h)))
            else: 
                return []
        for i in range(len(coords) - 1):
            cv2.line(img, coords[i], coords[i+1], color, thickness)
        for x, y in coords:
            cv2.circle(img, (x, y), 5, color, -1)
        return coords

    def analyze_ready(self):
        kf = int(self.keyframes.get('ready', -1))
        if kf == -1: 
            return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: 
            return 0

        scores = []
        ang = self.calc_angle(lms['r_wr'], lms['r_el'], lms['l_wr'])
        if 18 <= ang <= 70:
            s_ang = 100
        elif ang > 70:
            s_ang = max(10, 100 - (int((ang - 70)/10) + 1)*10)
        else:
            s_ang = max(10, 100 - (int((18 - ang)/2) + 1)*10)
            
        self.report['details']['Ready']['Arm_Angle'] = {
            "measured": round(ang, 2), 
            "target": "18 ~ 70", 
            "score": int(s_ang)
        }
        scores.append(s_ang)

        h_diff = lms['l_wr'].y - lms['l_sh'].y
        s_height = 100 if h_diff < 0 else max(10, 100 - (int(h_diff / 0.05) + 1) * 10)
        self.report['details']['Ready']['Left_Wrist_Height'] = {
            "measured": round(h_diff, 4), 
            "score": int(s_height)
        }
        scores.append(s_height)

        sh_w = abs(lms['r_sh'].x - lms['l_sh'].x)
        ft_w = abs(lms['r_ank'].x - lms['l_ank'].x)
        s_stance = 100 if ft_w > sh_w else max(10, 100 - (int((sh_w - ft_w) / 0.02) + 1) * 10)
        self.report['details']['Ready']['Stance_Width'] = {
            "measured": round(ft_w, 4), 
            "score": int(s_stance)
        }
        scores.append(s_stance)

        ratio = self.calculate_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        t_min, t_max = -0.5, 2.0
        s_wr_h = 100 if t_min <= ratio <= t_max else max(10, 100 - (int(min(abs(ratio-t_min), abs(ratio-t_max)) / 0.05) + 1) * 10)
        self.report['details']['Ready']['Wrist_Height_Ratio'] = {
            "measured": round(ratio, 2), 
            "score": int(s_wr_h)
        }
        scores.append(s_wr_h)

        cv2.imwrite(os.path.join(self.output_dir, "1_Ready.jpg"), frame)
        return sum(scores) / len(scores)

    def analyze_swing_sequence(self):
        ready_f = int(self.keyframes.get('ready', -1))
        back_f = int(self.keyframes.get('backswing', -1))
        impact_f = int(self.keyframes.get('impact', -1))
        
        if ready_f == -1 or impact_f == -1: 
            return 0

        _, rdy_lms = self._get_frame_and_landmarks(ready_f)
        _, imp_lms = self._get_frame_and_landmarks(impact_f)
        
        rot_scores, bs_scores = [], []
        
        if rdy_lms and imp_lms:
            hip_x_diff = abs(imp_lms['r_hip'].x - imp_lms['l_hip'].x)
            s_hip = 100 if hip_x_diff <= 0.03 else max(10, 100 - int((hip_x_diff - 0.03) * 500))
            self.report['details']['Rotation']['Hip_Frontal_Alignment'] = {
                "score": int(s_hip)
            }
            rot_scores.append(s_hip)

            init_sh_w = abs(rdy_lms['r_sh'].x - rdy_lms['l_sh'].x)
            curr_sh_w = abs(imp_lms['r_sh'].x - imp_lms['l_sh'].x)
            w_ratio = curr_sh_w / init_sh_w if init_sh_w != 0 else 1.0
            
            s_sh = 100 if (0.4 <= w_ratio <= 0.7) else max(10, 100 - int(min(abs(w_ratio - 0.4), abs(w_ratio - 0.7)) * 150))
            self.report['details']['Rotation']['Shoulder_Frontal_Alignment'] = {
                "score": int(s_sh)
            }
            rot_scores.append(s_sh)

        bs_lms = None
        if back_f != -1:
            _, bs_lms = self._get_frame_and_landmarks(back_f)

        if bs_lms:
            wx_diff = bs_lms['r_wr'].x - bs_lms['nose'].x
            s_wx = 100 if wx_diff < 0 else max(10, 100 - (int(wx_diff / 0.02) + 1) * 10)
            self.report['details']['Backswing']['Wrist_X_Depth'] = {
                "score": int(s_wx)
            }
            bs_scores.append(s_wx)

            e_ratio = self.calculate_ratio(bs_lms['r_el'], bs_lms['r_sh'], bs_lms['l_sh'])
            t_lift_min, t_lift_max = 1.5, 3.0
            
            if t_lift_min <= e_ratio <= t_lift_max:
                s_lift = 100
            else:
                diff_lift = min(abs(e_ratio - t_lift_min), abs(e_ratio - t_lift_max))
                s_lift = max(0, 100 - int(diff_lift * 200))
                
            self.report['details']['Backswing']['Elbow_Lift'] = {
                "score": int(s_lift)
            }
            bs_scores.append(s_lift)

            bs_ang = self.calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr'])
            t_ang_min, t_ang_max = 60, 110
            
            if t_ang_min <= bs_ang <= t_ang_max:
                s_bs_ang = 100
            else:
                diff_ang = min(abs(bs_ang - t_ang_min), abs(bs_ang - t_ang_max))
                s_bs_ang = max(0, 100 - int(diff_ang * 2.0))
                
            self.report['details']['Backswing']['L_Shape_Angle'] = {
                "score": int(s_bs_ang)
            }
            bs_scores.append(s_bs_ang)

        if back_f != -1:
            seq_2_f = int(ready_f + (back_f - ready_f) * 2 / 3)
            bi_step = max(1, (impact_f - back_f) // 3)
            frames_map = {
                "Seq_1_Ready": ready_f, 
                "Seq_2_Takeaway": seq_2_f, 
                "Seq_3_Backswing": back_f,
                "Seq_4_Downswing_1": back_f + bi_step, 
                "Seq_5_Downswing_2": back_f + (bi_step * 2), 
                "Seq_6_Impact": impact_f
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
            frame, _ = self._get_frame_and_landmarks(f_idx)
            if frame is not None:
                cv2.imwrite(os.path.join(self.output_dir, f"{name}.jpg"), frame)

        avg_rot = sum(rot_scores) / len(rot_scores) if rot_scores else 0
        avg_bs = sum(bs_scores) / len(bs_scores) if bs_scores else 0
        return (avg_rot + avg_bs) / 2

    def analyze_impact(self):
        kf = int(self.keyframes.get('impact', -1))
        if kf == -1: 
            return 0
        frame, lms = self._get_frame_and_landmarks(kf)
        if frame is None or not lms: 
            return 0
        
        scores = []
        ang = self.calc_angle(lms['r_sh'], lms['r_el'], lms['r_wr'])
        s_ang = 100 if 140 <= ang <= 180 else max(10, 100 - int(abs(140 - ang) * 2))
        self.report['details']['Impact']['Arm_Extension_Angle'] = {
            "score": int(s_ang)
        }
        scores.append(s_ang)

        w_ratio = self.calculate_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        t_min, t_max = 2.5, 4.5
        s_wr = 100 if t_min <= w_ratio <= t_max else max(10, 100 - (int(min(abs(w_ratio-t_min), abs(w_ratio-t_max)) / 0.5) + 1) * 10)
        self.report['details']['Impact']['Wrist_Height_Ratio'] = {
            "score": int(s_wr)
        }
        scores.append(s_wr)
        
        cv2.imwrite(os.path.join(self.output_dir, "3_Impact.jpg"), frame)
        return sum(scores) / len(scores)

    def analyze_follow(self):
        start_f = int(self.keyframes.get('impact', -1))
        if start_f == -1: 
            return 0
        
        end_f = min(start_f + 40, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        w, h = int(self.cap.get(3)), int(self.cap.get(4))
        fps = self.cap.get(5)
        
        out = cv2.VideoWriter(
            os.path.join(self.output_dir, "4_FollowSwing.mp4"), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (w, h)
        )
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        wrist_dropped = False
        final_wr_x, final_el_x = 0.0, 0.0

        for i in range(start_f, end_f + 1):
            ret, frame = self.cap.read()
            if not ret: 
                break
            
            res = self.pose_video.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = {name: res.pose_landmarks.landmark[idx] for name, idx in MP_MAP.items()}
                
                if lms['r_wr'].y > lms['r_el'].y: 
                    wrist_dropped = True
                
                final_wr_x = lms['r_wr'].x
                final_el_x = lms['r_el'].x

            out.write(frame)
        out.release()
        
        TOLERANCE = 0.03 
        
        if not wrist_dropped:
            score = 0
        elif final_wr_x > (final_el_x + TOLERANCE):
            score = 50
        else:
            score = 100

        self.report['details']['FollowSwing']['Performance'] = {
            "score": score
        }
        return score

    def run(self):
        if not self.cap.isOpened(): 
            return
        
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
            
        print(f"\n✅ 최종 분석 완료!")
        self.cap.release()
        self.pose_static.close()
        self.pose_video.close()