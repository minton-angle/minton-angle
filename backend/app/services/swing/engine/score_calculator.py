import os
import json
import math
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from typing import Dict, Optional

MP_MAP = {
    'nose': 0, 'r_sh': 12, 'r_el': 14, 'r_wr': 16, 'r_fin': 20,
    'l_sh': 11, 'l_el': 13, 'l_wr': 15, 'r_hip': 24, 'l_hip': 23,
    'r_ank': 28, 'l_ank': 27
}


class ScoreCalculator:
    """GolfAnalyzer 기반 점수 계산 + 이미지 저장 통합 엔진"""

    _initialized = False

    def __init__(self, gt_json_path: str = None):
        self.gt = None
        if gt_json_path and os.path.exists(gt_json_path):
            with open(gt_json_path, 'r', encoding='utf-8') as f:
                self.gt = json.load(f)

        if not ScoreCalculator._initialized:
            if self.gt:
                print("✅ [ScoreCalculator] 전문가 기준 로드 완료")
            else:
                print("⚠️ [ScoreCalculator] GT 파일 없음, 기본 임계값 사용")
            ScoreCalculator._initialized = True

        self.pose_static = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.pose_video = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )

    # ------------------------------------------------------------------ #
    #  수학 유틸리티
    # ------------------------------------------------------------------ #
    def _calc_angle(self, p1, p2, p3) -> float:
        a, b, c = np.array([p1.x, p1.y]), np.array([p2.x, p2.y]), np.array([p3.x, p3.y])
        ba, bc = a - b, c - b
        n = np.linalg.norm(ba) * np.linalg.norm(bc)
        if n == 0:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))))

    def _calc_ratio(self, target, sh_r, sh_l) -> float:
        sh_w = abs(sh_r.x - sh_l.x)
        if sh_w == 0:
            return 0.0
        sh_y_avg = (sh_r.y + sh_l.y) / 2
        return (sh_y_avg - target.y) / sh_w

    def _calc_angle_from_horizontal(self, center, target) -> float:
        """수평 기준 각도 계산 (FollowSwing용)"""
        dy = target.y - center.y
        dx = target.x - center.x
        if dx == 0:
            return 90.0
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    def _calc_rotation_score(self, min_diff: float, init_diff: float) -> float:
        """회전 점수 계산 (많이 돌수록 높은 점수)"""
        if init_diff == 0:
            return 100
        ratio = min_diff / init_diff
        if ratio <= 0.2:   return 100
        elif ratio <= 0.4: return 80
        elif ratio <= 0.6: return 60
        elif ratio <= 0.8: return 40
        elif ratio <= 1.0: return 20
        else:              return 0

    # ------------------------------------------------------------------ #
    #  영상에서 특정 프레임 + 랜드마크 추출 (EXIF 회전 보정 포함)
    # ------------------------------------------------------------------ #
    def _get_frame_and_landmarks(self, cap, frame_idx: int):
        # 주변 프레임을 탐색하여 가장 정확한 포즈를 찾음 (오프셋 로직)
        for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
            target = int(frame_idx) + offset
            if target < 0:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue

            # 🌟 [회전 보정] 핸드폰 세로 영상이 가로로 누워있는 경우 세워줌
            h, w = frame.shape[:2]
            if w > h:
                # 가로가 더 길다면 시계방향으로 90도 회전
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # 미디어파이프 분석 (RGB 변환 필수)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_static.process(img_rgb)

            # 랜드마크가 감지된 경우에만 반환
            if results.pose_landmarks:
                lms = {}
                for name, idx in MP_MAP.items():
                    lms[name] = results.pose_landmarks.landmark[idx]
                
                print(f"✅ 분석 성공: idx={target}, offset={offset}, shape={frame.shape}")
                return frame, lms

        # 🌟 모든 오프셋에서 랜드마크 추출 실패 시
        print(f"⚠️ 랜드마크 추출 실패: frame_idx={frame_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        
        # 실패하더라도 이미지는 똑바로 세워서 반환 (그래야 엑박이 안 뜸)
        if ret:
            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            return frame, {}
            
        return None, {}

    # ------------------------------------------------------------------ #
    #  시각화 유틸리티
    # ------------------------------------------------------------------ #
    def _draw_text(self, img, text, pos, color=(0, 0, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def _draw_chain(self, img, lms, points_chain, color=(0, 0, 255), thickness=3):
        h, w = img.shape[:2]
        coords = []
        for name in points_chain:
            if name not in lms:
                return []
            coords.append((int(lms[name].x * w), int(lms[name].y * h)))
        for i in range(len(coords) - 1):
            cv2.line(img, coords[i], coords[i + 1], color, thickness)
        for x, y in coords:
            cv2.circle(img, (x, y), 5, color, -1)
        return coords

    # ------------------------------------------------------------------ #
    #  Phase 1 – Ready
    # ------------------------------------------------------------------ #
    def _analyze_ready(self, cap, kf: int, output_dir: str, details: dict) -> float:
        frame, lms = self._get_frame_and_landmarks(cap, kf)
        if frame is None or not lms:
            return 0.0

        scores = []

        # 팔 각도 (패널티 *20으로 강화)
        ang = self._calc_angle(lms['r_wr'], lms['r_el'], lms['l_wr'])
        s = 100 if 18 <= ang <= 70 else max(10, 100 - (int((ang - 70) / 10) + 1) * 20 if ang > 70 else 100 - (int((18 - ang) / 2) + 1) * 20)
        details['Ready']['Arm_Angle'] = {
            "measured": round(ang, 2), "target": "18~70",
            "diff": round(ang - 70 if ang > 70 else (18 - ang if ang < 18 else 0), 2),
            "score": int(s)
        }
        scores.append(s)

        # 왼손 높이 (패널티 *20으로 강화)
        h_diff = lms['l_wr'].y - lms['l_sh'].y
        s2 = 100 if h_diff < 0 else max(10, 100 - (int(h_diff / 0.05) + 1) * 20)
        details['Ready']['Left_Wrist_Height'] = {
            "measured": round(h_diff, 4), "target": "<0",
            "diff": round(max(0, h_diff), 4),
            "score": int(s2)
        }
        scores.append(s2)

        # 스탠스 너비 (패널티 *20으로 강화)
        sh_w = abs(lms['r_sh'].x - lms['l_sh'].x)
        ft_w = abs(lms['r_ank'].x - lms['l_ank'].x)
        s3 = 100 if ft_w > sh_w else max(10, 100 - (int((sh_w - ft_w) / 0.02) + 1) * 20)
        details['Ready']['Stance_Width'] = {
            "measured": round(ft_w, 4), "target": f">{round(sh_w, 4)}",
            "diff": round(max(0, sh_w - ft_w), 4),
            "score": int(s3)
        }
        scores.append(s3)

        # 손목 높이 비율 (패널티 *20으로 강화)
        ratio = self._calc_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        s4 = 100 if -0.5 <= ratio <= 2.0 else max(10, 100 - (int(min(abs(ratio + 0.5), abs(ratio - 2.0)) / 0.05) + 1) * 20)
        details['Ready']['Wrist_Height_Ratio'] = {
            "measured": round(ratio, 2), "target": "-0.5~2.0",
            "diff": round(min(abs(ratio + 0.5), abs(ratio - 2.0)) if s4 < 100 else 0, 2),
            "score": int(s4)
        }
        scores.append(s4)

        # 이미지 저장
        coords = self._draw_chain(frame, lms, ['r_wr', 'r_el', 'l_wr'], color=(0, 0, 255))
        if coords and len(coords) >= 2:
            ex, ey = coords[1]
            self._draw_text(frame, f"{int(ang)} deg", (ex + 10, ey))
        cv2.imwrite(os.path.join(output_dir, "1_Ready.jpg"), frame)
        print(f"✅ DB 등록: READY → 1_Ready.jpg")

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------ #
    #  Phase 2 – Swing Sequence (Rotation + Backswing)
    # ------------------------------------------------------------------ #
    def _analyze_swing_sequence(self, cap, keyframes: dict, output_dir: str, details: dict) -> float:
        ready_f  = int(keyframes.get('ready',     -1))
        back_f   = int(keyframes.get('backswing', -1))
        impact_f = int(keyframes.get('impact',    -1))

        if ready_f == -1 or impact_f == -1:
            return 0.0

        _, rdy_lms = self._get_frame_and_landmarks(cap, ready_f)

        rot_scores, bs_scores = [], []

        # --- Rotation: ready~impact 전체 프레임 스캔 ---
        init_hip_diff, init_sh_diff = 0.1, 0.2
        if rdy_lms:
            init_hip_diff = abs(rdy_lms['r_hip'].x - rdy_lms['l_hip'].x)
            init_sh_diff  = abs(rdy_lms['r_sh'].x  - rdy_lms['l_sh'].x)

        min_hip_diff = 1.0
        min_sh_diff  = 1.0

        for f in range(ready_f, impact_f + 1):
            _, lms = self._get_frame_and_landmarks(cap, f)
            if lms:
                h_diff = abs(lms['r_hip'].x - lms['l_hip'].x)
                s_diff = abs(lms['r_sh'].x  - lms['l_sh'].x)
                if h_diff < min_hip_diff: min_hip_diff = h_diff
                if s_diff < min_sh_diff:  min_sh_diff  = s_diff

        s_hip = self._calc_rotation_score(min_hip_diff, init_hip_diff)
        s_sh  = self._calc_rotation_score(min_sh_diff,  init_sh_diff)

        details['Rotation']['Hip_Max_Rotation'] = {
            "initial_diff": round(init_hip_diff, 4),
            "min_diff":     round(min_hip_diff, 4),
            "target_msg":   "0 (Fully Crossed)",
            "score":        int(s_hip)
        }
        details['Rotation']['Shoulder_Max_Rotation'] = {
            "initial_diff": round(init_sh_diff, 4),
            "min_diff":     round(min_sh_diff, 4),
            "target_msg":   "0 (Fully Crossed)",
            "score":        int(s_sh)
        }
        rot_scores.extend([s_hip, s_sh])

        # --- Backswing ---
        bs_lms = None
        is_backswing_valid = (back_f != -1 and ready_f < back_f < impact_f)
        if is_backswing_valid:
            _, bs_lms = self._get_frame_and_landmarks(cap, back_f)

        if bs_lms:
            e_ratio = self._calc_ratio(bs_lms['r_el'], bs_lms['r_sh'], bs_lms['l_sh'])
            diff_lift = min(abs(e_ratio - 1.5), abs(e_ratio - 3.0))
            s_lift = 100 if 1.5 <= e_ratio <= 3.0 else max(0, 100 - int(diff_lift * 400))
            details['Backswing']['Elbow_Lift'] = {
                "measured": round(e_ratio, 2), "target": "1.5~3.0", "score": int(s_lift)
            }
            bs_scores.append(s_lift)

            # ⭐ 0점 연쇄 처리
            if s_lift == 0:
                details['Backswing']['Wrist_X_Depth'] = {
                    "measured": round(bs_lms['r_wr'].x - bs_lms['nose'].x, 4),
                    "target": "0 (Failed: Elbow Lift is 0)", "score": 0
                }
                details['Backswing']['L_Shape_Angle'] = {
                    "measured": round(self._calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr']), 2),
                    "target": "0 (Failed: Elbow Lift is 0)", "score": 0
                }
                bs_scores.extend([0, 0])
            else:
                wx_diff = bs_lms['r_wr'].x - bs_lms['nose'].x
                s_wx = 100 if wx_diff < 0 else max(0, 100 - (int(wx_diff / 0.02) + 1) * 20)
                details['Backswing']['Wrist_X_Depth'] = {
                    "measured": round(wx_diff, 4), "target": "< 0", "score": int(s_wx)
                }
                bs_scores.append(s_wx)

                bs_ang = self._calc_angle(bs_lms['r_sh'], bs_lms['r_el'], bs_lms['r_wr'])
                diff_ang = min(abs(bs_ang - 60), abs(bs_ang - 110))
                s_bs_ang = 100 if 60 <= bs_ang <= 110 else max(0, 100 - int(diff_ang * 4.0))
                details['Backswing']['L_Shape_Angle'] = {
                    "measured": round(bs_ang, 2), "target": "60~110", "score": int(s_bs_ang)
                }
                bs_scores.append(s_bs_ang)

        else:
            # 백스윙 프레임 없으면 3개 지표 모두 0점 강제 합산
            print("⚠️ 백스윙 프레임이 유효하지 않아 백스윙 3개 지표를 모두 0점으로 처리합니다.")
            details['Backswing']['Elbow_Lift']    = {"measured": 0, "target": "1.5~3.0", "score": 0}
            details['Backswing']['Wrist_X_Depth'] = {"measured": 0, "target": "< 0",     "score": 0}
            details['Backswing']['L_Shape_Angle'] = {"measured": 0, "target": "60~110",  "score": 0}
            bs_scores.extend([0, 0, 0])

        # --- 시퀀스 이미지 저장 ---
        if is_backswing_valid:
            seq_2_f = int(ready_f + (back_f - ready_f) * 2 / 3)
            bi_step = max(1, (impact_f - back_f) // 3)
            frames_map = {
                "Seq_1_Ready":      ready_f,
                "Seq_2_Takeaway":   seq_2_f,
                "Seq_3_Backswing":  back_f,
                "Seq_4_Downswing_1": back_f + bi_step,
                "Seq_5_Downswing_2": back_f + bi_step * 2,
                "Seq_6_Impact":     impact_f
            }
        else:
            step = max(1, (impact_f - ready_f) // 5)
            frames_map = {
                "Seq_1_Ready":      ready_f,
                "Seq_2_Takeaway":   ready_f + step,
                "Seq_3_Backswing":  ready_f + step * 2,
                "Seq_4_Downswing_1": ready_f + step * 3,
                "Seq_5_Downswing_2": ready_f + step * 4,
                "Seq_6_Impact":     impact_f
            }

        for name, f_idx in frames_map.items():
            frame, lms = self._get_frame_and_landmarks(cap, f_idx)
            if frame is not None and lms:
                self._draw_chain(frame, lms, ['r_wr', 'r_el', 'r_sh'], color=(0, 255, 0))
                self._draw_chain(frame, lms, ['l_hip', 'r_hip'],        color=(0, 0, 255))
                self._draw_chain(frame, lms, ['l_sh', 'r_sh'],          color=(255, 0, 0))
            if frame is not None:
                cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), frame)
                print(f"✅ DB 등록: {name.upper()} → {name}.jpg")

        avg_rot = sum(rot_scores) / len(rot_scores) if rot_scores else 0
        avg_bs  = sum(bs_scores)  / len(bs_scores)  if bs_scores  else 0
        return (avg_rot + avg_bs) / 2

    # ------------------------------------------------------------------ #
    #  Phase 3 – Impact
    # ------------------------------------------------------------------ #
    def _analyze_impact(self, cap, kf: int, output_dir: str, details: dict) -> float:
        frame, lms = self._get_frame_and_landmarks(cap, kf)
        if frame is None or not lms:
            return 0.0

        scores = []

        # 팔꿈치 펴짐 (패널티 *4로 강화)
        ang = self._calc_angle(lms['r_sh'], lms['r_el'], lms['r_wr'])
        s = 100 if 140 <= ang <= 180 else max(10, 100 - int(abs(140 - ang) * 4))
        details['Impact']['Arm_Extension_Angle'] = {
            "measured": round(ang, 2), "target": "140~180",
            "diff": round(max(0, 140 - ang), 2),
            "score": int(s)
        }
        scores.append(s)

        # 손목 높이 비율 (패널티 *20으로 강화)
        w_ratio = self._calc_ratio(lms['r_wr'], lms['r_sh'], lms['l_sh'])
        diff = min(abs(w_ratio - 2.5), abs(w_ratio - 4.5))
        s2 = 100 if 2.5 <= w_ratio <= 4.5 else max(10, 100 - (int(diff / 0.5) + 1) * 20)
        details['Impact']['Wrist_Height_Ratio'] = {
            "measured": round(w_ratio, 2), "target": "2.5~4.5",
            "diff": round(diff if s2 < 100 else 0, 2),
            "score": int(s2)
        }
        scores.append(s2)

        coords = self._draw_chain(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 0, 255))
        if coords and len(coords) >= 2:
            ex, ey = coords[1]
            self._draw_text(frame, f"{int(ang)} deg", (ex, ey - 20))
        cv2.imwrite(os.path.join(output_dir, "3_Impact.jpg"), frame)
        print(f"✅ DB 등록: IMPACT → 3_Impact.jpg")

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------ #
    #  Phase 4 – Follow Swing (영상 저장)
    # ------------------------------------------------------------------ #
    def _analyze_follow(self, cap, kf: int, output_dir: str, details: dict) -> float:
        if kf == -1:
            return 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_f = min(kf + 40, total_frames - 1)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path  = os.path.join(output_dir, "4_FollowSwing.mp4")
        temp_path = os.path.join(output_dir, "4_FollowSwing_temp.avi")
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)

        # ⭐ 팀원 알고리즘: 각도 30도 + X 교차 두 조건
        has_reached_angle_30 = False
        has_matched_x        = False
        prev_x_diff          = None

        for _ in range(kf, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            res = self.pose_video.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = {name: res.pose_landmarks.landmark[idx] for name, idx in MP_MAP.items()}
                self._draw_chain(frame, lms, ['r_sh', 'r_el', 'r_wr'], color=(0, 0, 255), thickness=3)

                # 조건 1: 수평 각도 30도 돌파
                angle_horiz = self._calc_angle_from_horizontal(lms['r_sh'], lms['r_wr'])
                if angle_horiz > 30:
                    has_reached_angle_30 = True

                # 조건 2: 손목-팔꿈치 X좌표 일치 또는 교차
                curr_x_diff = lms['r_wr'].x - lms['r_el'].x
                if abs(curr_x_diff) <= 0.02:
                    has_matched_x = True
                if prev_x_diff is not None and (prev_x_diff * curr_x_diff <= 0):
                    has_matched_x = True
                prev_x_diff = curr_x_diff

                self._draw_text(
                    frame, f"Ang: {int(angle_horiz)}", (50, 50),
                    color=(0, 255, 0) if angle_horiz > 30 else (0, 0, 255)
                )
            out.write(frame)
        out.release()

        # ffmpeg H.264 변환
        import subprocess
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', out_path
        ], capture_output=True)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"✅ DB 등록: FOLLOWSWING → 4_FollowSwing.mp4")

        if not has_reached_angle_30:
            score, msg = 0,   "Follow Swing Too Low (Angle < 30)"
        elif not has_matched_x:
            score, msg = 50,  "Angle > 30 but Wrist and Elbow X did not match"
        else:
            score, msg = 100, "Perfect Follow Swing (Angle > 30 & X Matched)"

        details['FollowSwing']['Performance'] = {
            "score":            score,
            "message":          msg,
            "has_reached_30":   has_reached_angle_30,
            "has_matched_x":    has_matched_x
        }
        return float(score)

    # ------------------------------------------------------------------ #
    #  Public API – evaluate_user
    # ------------------------------------------------------------------ #
    def evaluate_user(
        self,
        df: pd.DataFrame,
        keyframes: Dict,
        video_path: str,
        output_dir: str
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)

        details = {
            "Ready": {}, "Rotation": {}, "Backswing": {},
            "Impact": {}, "FollowSwing": {}
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 영상 열기 실패: {video_path}")
            return {"total_score": 0, "details": details}

        try:
            kf_ready  = int(keyframes.get('ready',  -1))
            kf_impact = int(keyframes.get('impact', -1))

            print("--- [1] Ready Phase ---")
            s1 = self._analyze_ready(cap, kf_ready, output_dir, details)

            print("--- [2] Swing Sequence ---")
            s2 = self._analyze_swing_sequence(cap, keyframes, output_dir, details)

            print("--- [3] Impact Phase ---")
            s3 = self._analyze_impact(cap, kf_impact, output_dir, details)

            print("--- [4] Follow Swing ---")
            s4 = self._analyze_follow(cap, kf_impact, output_dir, details)

        finally:
            cap.release()

        s1 = s1 if s1 is not None else 0
        s2 = s2 if s2 is not None else 0
        s3 = s3 if s3 is not None else 0
        s4 = s4 if s4 is not None else 0

        total = round((s1 + s2 + s3 + s4) / 4, 1)

        return {
            "total_score": total,
            "details":     details
        }