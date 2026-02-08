import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- ⚙️ 설정 ---
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
        
        # ⚠️ 만약 JSON 파일이 없을 때를 대비한 '기본 백핸드 범위' (안전장치)
        self.default_ranges = {
            "thumb_press": {"min": 0.0, "max": 20.0},      # 엄지는 펴져야 함 (0에 가까울수록 좋음)
            "index_support": {"min": 120.0, "max": 170.0}, # 검지는 받쳐주는 형태
            "ti_gap": {"min": 20.0, "max": 45.0}           # 적당한 벌림
        }
        
        # 파일 로드 시도
        self.ranges = self.default_ranges.copy()
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Q1(하위 25%) ~ Q3(상위 25%) 범위를 기준으로 잡음 (너무 빡빡하면 min/max로 변경 가능)
                    self.ranges["thumb_press"] = {"min": data["thumb_press"]["q1"], "max": data["thumb_press"]["q3"]}
                    self.ranges["index_support"] = {"min": data["index_support"]["q1"], "max": data["index_support"]["q3"]}
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
        """범위 벗어난 만큼의 오차 반환 (0이면 통과)"""
        if value < target_range['min']:
            return target_range['min'] - value
        elif value > target_range['max']:
            return value - target_range['max']
        else:
            return 0.0

    def evaluate(self, landmarks):
        lm = landmarks.landmark
        
        # --- 1. 실시간 각도 계산 (백핸드 로직) ---
        
        # (1) Thumb Press: 엄지 펴짐 (2-3-4)
        curr_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # (2) Index Support: 검지 지지 (5-6-7)
        curr_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
            
        # (3) TI Gap: 엄지-검지 간격 (0-2, 0-5)
        curr_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))

        # --- 2. 오차(Penalty) 계산 ---
        diff_thumb = self.check_range(curr_thumb, self.ranges['thumb_press'])
        diff_index = self.check_range(curr_index, self.ranges['index_support'])
        diff_gap = self.check_range(curr_gap, self.ranges['ti_gap'])

        # --- 3. 점수 산정 (가중치 적용) ---
        # 백핸드는 엄지가 제일 중요하므로 가중치 2.0 배
        penalty = (diff_thumb * 2.0) + (diff_index * 1.0) + (diff_gap * 1.0)
        score = max(0, 100 - int(penalty))

        # --- 4. 화면 표시용 데이터 ---
        status_info = [
            {"label": "Thumb Press",  "val": curr_thumb, "range": self.ranges['thumb_press'], "pass": diff_thumb == 0},
            {"label": "Index Support","val": curr_index, "range": self.ranges['index_support'], "pass": diff_index == 0},
            {"label": "TI Gap",       "val": curr_gap,   "range": self.ranges['ti_gap'],      "pass": diff_gap == 0},
        ]

        # --- 5. 실시간 피드백 메시지 ---
        feedbacks = []
        
        # 엄지 피드백
        if diff_thumb > 0:
            if curr_thumb > self.ranges['thumb_press']['max']:
                feedbacks.append("👍 Straighten Thumb! (Push)") # 엄지 펴서 눌러라
            else:
                feedbacks.append("👍 Check Thumb")

        # 검지 피드백
        if diff_index > 0:
            feedbacks.append("☝️ Adjust Index Finger")

        # 간격 피드백
        if diff_gap > 0:
            if curr_gap > self.ranges['ti_gap']['max']:
                feedbacks.append("🤏 Gap too wide")
            else:
                feedbacks.append("🖐 Gap too narrow")

        if score >= 95: feedbacks = ["🏆 Perfect Backhand!"]
        elif not feedbacks: feedbacks.append("✅ Maintain Grip")

        return score, feedbacks, status_info

    def run(self):
        cap = cv2.VideoCapture(CAM_ID)
        
        print("🏸 백핸드 코칭 시작! (ESC를 누르면 종료)")
        
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
                    
                    # 1. 종합 점수 (Score)
                    score_color = (0, 255, 0) if score >= 90 else (0, 255, 255) if score >= 70 else (0, 0, 255)
                    cv2.putText(image, f"Score: {score}", (20, y_start), cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2)
                    
                    # 2. 상세 지표 표시
                    for i, item in enumerate(status_list):
                        text_color = (0, 255, 0) if item['pass'] else (0, 0, 255)
                        
                        # 예: "Thumb Press : 12 [0~15]"
                        r_min = int(item['range']['min'])
                        r_max = int(item['range']['max'])
                        display_text = f"{item['label']} : {int(item['val'])} [{r_min}~{r_max}]"
                        
                        cv2.putText(image, display_text, (20, y_start + 40 + (i * 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    # 3. 피드백 메시지 (화면 하단에 강조)
                    for i, msg in enumerate(feedbacks):
                        cv2.putText(image, msg, (20, y_start + 160 + (i * 40)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            else:
                cv2.putText(image, "Show Backhand Grip 🏸", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)

            cv2.imshow('Badminton Backhand Coach', image)
            if cv2.waitKey(5) & 0xFF == 27: break # ESC 키
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = BackhandCoach(GT_FILE)
    coach.run()