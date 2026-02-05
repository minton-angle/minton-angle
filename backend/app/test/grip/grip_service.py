import cv2
import mediapipe as mp
import numpy as np

# --- 설정 ---
CAM_ID = 0

class RealTimeCoachSingleFile:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # ✅ 여기에 JSON 데이터를 직접 입력했습니다 (Q1 ~ Q3 범위)
        # 제공해주신 데이터 값을 그대로 매핑했습니다.
        self.ranges = {
            "v_shape": {
                "min": 19.69,  # q1
                "max": 25.65   # q3
            },
            "thumb": {
                "min": 118.26, # q1
                "max": 162.90  # q3
            },
            "trigger_pip": {   # trigger_567
                "min": 123.30, # q1
                "max": 149.19  # q3
            },
            "trigger_dip": {   # trigger_678
                "min": 88.49,  # q1
                "max": 134.24  # q3
            }
        }
        
        print("✅ 그립 범위 데이터 로드 완료 (내장 데이터 사용)")
        print(f"🎯 적용 범위: {self.ranges}")

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
        if value < target_range['min']:
            return target_range['min'] - value
        elif value > target_range['max']:
            return value - target_range['max']
        else:
            return 0.0

    def evaluate(self, landmarks):
        lm = landmarks.landmark
        
        # 1. 각도 계산 (Raw Angle 사용)
        curr_v = self.calculate_angle(self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))
        curr_thumb = self.calculate_angle(self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
        curr_pip = self.calculate_angle(self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
        curr_dip = self.calculate_angle(self.get_vector(lm[7], lm[6]), self.get_vector(lm[7], lm[8]))

        # 2. 오차 계산 (Diff)
        diff_v = self.check_range(curr_v, self.ranges['v_shape'])
        diff_thumb = self.check_range(curr_thumb, self.ranges['thumb'])
        diff_pip = self.check_range(curr_pip, self.ranges['trigger_pip'])
        diff_dip = self.check_range(curr_dip, self.ranges['trigger_dip'])

        # 3. 점수 계산 (엄격한 가중치 적용)
        penalty = (diff_v * 5.0) + (diff_thumb * 3.0) + (diff_pip * 3.0) + (diff_dip * 2.0)
        score = max(0, 100 - int(penalty))

        # 4. 화면 표시용 데이터 구성
        status_info = [
            {"label": "V-Shape", "val": curr_v, "min": self.ranges['v_shape']['min'], "max": self.ranges['v_shape']['max'], "pass": diff_v == 0},
            {"label": "Thumb",   "val": curr_thumb, "min": self.ranges['thumb']['min'], "max": self.ranges['thumb']['max'], "pass": diff_thumb == 0},
            {"label": "Idx PIP", "val": curr_pip, "min": self.ranges['trigger_pip']['min'], "max": self.ranges['trigger_pip']['max'], "pass": diff_pip == 0},
            {"label": "Idx DIP", "val": curr_dip, "min": self.ranges['trigger_dip']['min'], "max": self.ranges['trigger_dip']['max'], "pass": diff_dip == 0},
        ]

        # 5. 피드백 메시지 생성
        feedbacks = []
        if diff_v > 0: feedbacks.append("⚠️ Fix V-Shape")
        if diff_thumb > 0: feedbacks.append("👍 Fix Thumb")
        if diff_pip > 0: feedbacks.append("☝️ Fix Index(PIP)")
        if diff_dip > 0: feedbacks.append("☝️ Fix Index(DIP)")

        if score >= 95: feedbacks = ["🏆 World Class!"]
        elif not feedbacks: feedbacks.append("✅ Perfect Range")

        return score, feedbacks, status_info

    def run(self):
        cap = cv2.VideoCapture(CAM_ID)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            y_start = 60 # 텍스트 시작 높이

            if results.multi_hand_landmarks:
                for hand_landmarks, world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    score, feedbacks, status_list = self.evaluate(world_landmarks)

                    # --- UI 출력 ---
                    
                    # 1. 점수 표시 (90+: 초록, 70+: 노랑, 미만: 빨강)
                    score_color = (0, 255, 0) if score >= 90 else (0, 255, 255) if score >= 70 else (0, 0, 255)
                    cv2.putText(image, f"Score: {score}", (20, y_start), cv2.FONT_HERSHEY_DUPLEX, 1.3, score_color, 2)
                    
                    # 2. 상세 지표 (통과: 초록, 실패: 빨강)
                    for i, item in enumerate(status_list):
                        text_color = (0, 255, 0) if item['pass'] else (0, 0, 255)
                        
                        # "항목 : 현재값 [최소~최대]" 표시
                        display_text = f"{item['label']} : {int(item['val'])} [{int(item['min'])}~{int(item['max'])}]"
                        
                        cv2.putText(image, display_text, (20, y_start + 40 + (i * 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    # 3. 피드백 메시지
                    for i, msg in enumerate(feedbacks):
                        cv2.putText(image, msg, (20, y_start + 180 + (i * 35)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            else:
                cv2.putText(image, "Show Hand 🏸", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (150, 150, 150), 1)

            cv2.imshow('Badminton Grip Coach (One File)', image)
            if cv2.waitKey(5) & 0xFF == 27: break # ESC로 종료
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = RealTimeCoachSingleFile()
    coach.run()