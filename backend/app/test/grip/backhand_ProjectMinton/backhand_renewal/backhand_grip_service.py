import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- ⚙️ 설정 ---
# 3단계에서 만든 파일명 (4가지 지표가 들어있는 파일)
GT_FILE = "backhand_gt_range.json" 
CAM_ID = 0

class BackhandCoach:
    def __init__(self, gt_path):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # ⚠️ JSON 파일 없을 때를 대비한 '기본 안전장치'
        # (Thumb Press는 180도 기준, Grip Cross 추가됨)
        self.default_ranges = {
            "thumb_press":   {"min": 160.0, "max": 180.0}, # 엄지는 펴야 함 (일직선)
            "index_support": {"min": 130.0, "max": 170.0}, # 검지는 아치형
            "grip_cross":    {"min": 50.0,  "max": 90.0},  # 백핸드 구조 (수직)
            "ti_gap":        {"min": 20.0,  "max": 45.0}   # 손목 공간
        }
        
        # 파일 로드 로직
        self.ranges = self.default_ranges.copy()
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 파일에서 범위 읽어오기 (Q1 ~ Q3 사용)
                    self.ranges["thumb_press"] = {"min": data["thumb_press"]["q1"], "max": data["thumb_press"]["q3"]}
                    self.ranges["index_support"] = {"min": data["index_support"]["q1"], "max": data["index_support"]["q3"]}
                    self.ranges["grip_cross"] = {"min": data["grip_cross"]["q1"], "max": data["grip_cross"]["q3"]}
                    self.ranges["ti_gap"] = {"min": data["ti_gap"]["q1"], "max": data["ti_gap"]["q3"]}
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
        # 관용 범위 (Tolerance): 너무 빡빡하지 않게 ±5도 정도 여유 줌
        tolerance = 5.0
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
        
        # --- 1. 실시간 각도 계산 (4-Factor) ---
        
        # (1) Thumb Press: 180도에 가까울수록 일직선
        curr_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # (2) Index Support: 140~160도 아치형
        curr_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
            
        # (3) Grip Cross: 60~90도 (백핸드 구조 확인)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        curr_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        # (4) TI Gap: 20~40도 (손목 공간)
        curr_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))

        # --- 2. 오차(Penalty) 계산 ---
        diff_thumb = self.check_range(curr_thumb, self.ranges['thumb_press'])
        diff_index = self.check_range(curr_index, self.ranges['index_support'])
        diff_cross = self.check_range(curr_cross, self.ranges['grip_cross'])
        diff_gap   = self.check_range(curr_gap,   self.ranges['ti_gap'])

        # --- 3. 점수 산정 (가중치 적용) ---
        # Grip Cross(구조)와 Thumb Press(파워)가 틀리면 점수 대폭 깎임
        penalty = (diff_cross * 2.0) + (diff_thumb * 1.5) + (diff_index * 1.0) + (diff_gap * 0.8)
        score = max(0, 100 - int(penalty))

        # --- 4. 화면 표시용 데이터 ---
        status_info = [
            {"label": "Thumb Press",   "val": curr_thumb, "range": self.ranges['thumb_press'], "pass": diff_thumb == 0},
            {"label": "Index Support", "val": curr_index, "range": self.ranges['index_support'], "pass": diff_index == 0},
            {"label": "Grip Cross",    "val": curr_cross, "range": self.ranges['grip_cross'],  "pass": diff_cross == 0},
            {"label": "TI Gap",        "val": curr_gap,   "range": self.ranges['ti_gap'],      "pass": diff_gap == 0},
        ]

        # --- 5. 실시간 피드백 메시지 (우선순위별) ---
        feedbacks = []
        
        # [1순위] 구조 자체가 틀린 경우 (포핸드처럼 잡음)
        if diff_cross > 0:
            if curr_cross < self.ranges['grip_cross']['min']:
                feedbacks.append("⚠️ Not Backhand Grip! (Raise Thumb)") # 엄지를 더 세워라
        
        # [2순위] 엄지가 굽은 경우
        if diff_thumb > 0:
            if curr_thumb < self.ranges['thumb_press']['min']:
                feedbacks.append("👍 Straighten Thumb (Press Hard)") # 엄지 펴서 눌러라

        # [3순위] 검지 지지
        if diff_index > 0:
            if curr_index > self.ranges['index_support']['max']:
                 feedbacks.append("☝️ Curve Index Finger") # 너무 폈다
            else:
                 feedbacks.append("☝️ Relax Index Finger") # 너무 굽혔다

        # [4순위] 간격
        if diff_gap > 0:
            feedbacks.append("🖐 Adjust Gap")

        # 만점 피드백
        if score >= 90 and not feedbacks:
            feedbacks = ["🏆 Perfect Backhand!"]
        elif score >= 80 and not feedbacks:
            feedbacks = ["✅ Good Grip"]

        return score, feedbacks, status_info

    def run(self):
        cap = cv2.VideoCapture(CAM_ID)
        print("🏸 백핸드 코칭 시스템 가동! (종료: ESC)")
        
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
                    
                    # 평가 실행
                    score, feedbacks, status_list = self.evaluate(world_landmarks)

                    # --- UI 그리기 ---
                    
                    # 1. 점수 (Score)
                    score_color = (0, 255, 0) if score >= 90 else (0, 255, 255) if score >= 70 else (0, 0, 255)
                    cv2.putText(image, f"Score: {score}", (20, y_start), cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2)
                    
                    # 2. 상세 지표 (4대장)
                    for i, item in enumerate(status_list):
                        text_color = (0, 255, 0) if item['pass'] else (0, 0, 255)
                        
                        r_min = int(item['range']['min'])
                        r_max = int(item['range']['max'])
                        display_text = f"{item['label']} : {int(item['val'])} [{r_min}~{r_max}]"
                        
                        cv2.putText(image, display_text, (20, y_start + 40 + (i * 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    # 3. 피드백 메시지 (화면 하단)
                    for i, msg in enumerate(feedbacks):
                        # 중요도에 따라 색상 다르게 (Not Backhand는 빨강)
                        msg_color = (0, 0, 255) if "Not Backhand" in msg else (0, 255, 255)
                        cv2.putText(image, msg, (20, y_start + 180 + (i * 40)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)

            else:
                cv2.putText(image, "Show Backhand Grip 🏸", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)

            cv2.imshow('Badminton Backhand Coach', image)
            if cv2.waitKey(5) & 0xFF == 27: break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = BackhandCoach(GT_FILE)
    coach.run()