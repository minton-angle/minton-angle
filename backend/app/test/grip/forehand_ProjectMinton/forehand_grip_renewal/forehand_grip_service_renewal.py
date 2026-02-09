import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- ⚙️ 설정 ---
# 아까 만든 JSON 파일명과 똑같아야 합니다!
GT_FILE = "forehand_gt_range.json" 
CAM_ID = 0

class ForehandCoach:
    def __init__(self, gt_path):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # ⚠️ 파일 없을 때를 대비한 기본값 (안전장치)
        # (V-Shape, Thumb Flex, Trigger, Grip Cross)
        self.default_ranges = {
            "v_shape":     {"min": 20.0, "max": 50.0},
            "thumb_flex":  {"min": 10.0, "max": 40.0},
            "trigger":     {"min": 120.0, "max": 160.0},
            "grip_cross":  {"min": 0.0,   "max": 40.0}   # 🔥 포핸드는 낮아야 함!
        }
        
        # 파일 로드 로직
        self.ranges = self.default_ranges.copy()
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 파일에서 범위 읽어오기 (Q1 ~ Q3 사용)
                    # 키 이름이 조금 달라도 유연하게 처리
                    if "v_shape" in data:
                        self.ranges["v_shape"] = {"min": data["v_shape"]["q1"], "max": data["v_shape"]["q3"]}
                    if "thumb_flex" in data:
                        self.ranges["thumb_flex"] = {"min": data["thumb_flex"]["q1"], "max": data["thumb_flex"]["q3"]}
                    if "trigger" in data:
                        self.ranges["trigger"] = {"min": data["trigger"]["q1"], "max": data["trigger"]["q3"]}
                    if "grip_cross" in data:
                        self.ranges["grip_cross"] = {"min": data["grip_cross"]["q1"], "max": data["grip_cross"]["q3"]}
                        
                print(f"✅ 정답 데이터 로드 완료: {gt_path}")
            except Exception as e:
                print(f"⚠️ 파일 로드 실패 (기본값 사용): {e}")
        else:
            print("⚠️ 정답 파일이 없습니다. 기본값으로 코칭합니다.")

    def get_vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0
        dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def check_range(self, value, target_range):
        """범위 오차 계산 (0이면 통과)"""
        tolerance = 5.0 # 5도 정도는 봐줌
        min_limit = target_range['min'] - tolerance
        max_limit = target_range['max'] + tolerance

        if value < min_limit:
            return min_limit - value
        elif value > max_limit:
            return value - max_limit
        else:
            return 0.0

    def evaluate(self, landmarks):
        lm = landmarks.landmark
        
        # --- 1. 실시간 각도 계산 (포핸드 4대장) ---
        
        # (1) V-Shape
        curr_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))
            
        # (2) Thumb Flex (엄지 굽힘)
        curr_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # (3) Trigger (검지)
        curr_trigger = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))

        # (4) Grip Cross (교차각)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        curr_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        # --- 2. 오차(Penalty) 계산 ---
        diff_v = self.check_range(curr_v, self.ranges['v_shape'])
        diff_thumb = self.check_range(curr_thumb, self.ranges['thumb_flex'])
        diff_trigger = self.check_range(curr_trigger, self.ranges['trigger'])
        diff_cross = self.check_range(curr_cross, self.ranges['grip_cross'])

        # --- 3. 점수 산정 ---
        # Cross(구조)와 V-Shape(기본기)가 가장 중요함
        penalty = (diff_cross * 2.0) + (diff_v * 1.5) + (diff_thumb * 1.0) + (diff_trigger * 0.8)
        score = max(0, 100 - int(penalty))

        # --- 4. 화면 표시용 데이터 ---
        status_info = [
            {"label": "V-Shape",    "val": curr_v,       "range": self.ranges['v_shape'],    "pass": diff_v == 0},
            {"label": "Grip Cross", "val": curr_cross,   "range": self.ranges['grip_cross'], "pass": diff_cross == 0},
            {"label": "Thumb Flex", "val": curr_thumb,   "range": self.ranges['thumb_flex'], "pass": diff_thumb == 0},
            {"label": "Trigger",    "val": curr_trigger, "range": self.ranges['trigger'],    "pass": diff_trigger == 0},
        ]

        # --- 5. 실시간 피드백 메시지 ---
        feedbacks = []
        
        # [1순위] 구조 판별 (백핸드처럼 잡았는지 확인)
        if diff_cross > 0:
            if curr_cross > self.ranges['grip_cross']['max']:
                feedbacks.append("⚠️ Too Crossed! (Like Backhand)") 
                feedbacks.append("   -> Lay thumb down")

        # [2순위] V-Shape (악수)
        if diff_v > 0:
            if curr_v < self.ranges['v_shape']['min']:
                feedbacks.append("🤝 Widen V-Shape (Handshake)")

        # [3순위] 엄지 (감싸기)
        if diff_thumb > 0:
             if curr_thumb < self.ranges['thumb_flex']['min']: 
                 feedbacks.append("👍 Bend Thumb (Wrap it)")

        if score >= 90 and not feedbacks:
            feedbacks = ["🏆 Perfect Forehand!"]
        elif score >= 80 and not feedbacks:
            feedbacks = ["✅ Good Grip"]

        return score, feedbacks, status_info

    def run(self):
        cap = cv2.VideoCapture(CAM_ID)
        print("🏸 포핸드 코칭 시스템 가동! (종료: ESC)")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1) # 거울 모드
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            y_start = 50 

            if results.multi_hand_landmarks:
                for hand_landmarks, world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    score, feedbacks, status_list = self.evaluate(world_landmarks)

                    # --- UI 그리기 ---
                    score_color = (0, 255, 0) if score >= 90 else (0, 255, 255) if score >= 70 else (0, 0, 255)
                    cv2.putText(image, f"Score: {score}", (20, y_start), cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2)
                    
                    for i, item in enumerate(status_list):
                        text_color = (0, 255, 0) if item['pass'] else (0, 0, 255)
                        r_min = int(item['range']['min'])
                        r_max = int(item['range']['max'])
                        display_text = f"{item['label']} : {int(item['val'])} [{r_min}~{r_max}]"
                        cv2.putText(image, display_text, (20, y_start + 40 + (i * 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    for i, msg in enumerate(feedbacks):
                        # 중요도 색상 (빨강: 구조 오류, 노랑: 자세 수정)
                        msg_color = (0, 0, 255) if "Too Crossed" in msg else (0, 255, 255)
                        cv2.putText(image, msg, (20, y_start + 180 + (i * 40)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)
            else:
                cv2.putText(image, "Show Forehand Grip 🤝", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)

            cv2.imshow('Badminton Forehand Coach', image)
            if cv2.waitKey(5) & 0xFF == 27: break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = ForehandCoach(GT_FILE)
    coach.run()