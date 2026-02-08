import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- 설정 ---
GT_FILE = "grip_gt.json" # 1단계에서 만든 파일
THRESHOLD = 15.0         # 허용 오차 범위 (도)
CAM_ID = 0               # 웹캠 번호

class RealTimeCoach:
    def __init__(self, gt_path):
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # GT 데이터 로드
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                self.gt_data = json.load(f)
            print("GT 데이터를 성공적으로 로드했습니다.")
        else:
            print("경고: GT 파일이 없습니다. 기본값을 사용합니다.")
            self.gt_data = {"v_shape": 30.0, "thumb": 20.0, "trigger": 150.0}

    def get_vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0
        dot = np.clip(np.dot(v1/norm1, v2/norm2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def evaluate(self, landmarks):
        lm = landmarks.landmark
        
        # 현재 각도 계산
        # 1. V-Shape
        curr_v = self.calculate_angle(
            self.get_vector(lm[0], lm[1]), self.get_vector(lm[0], lm[5]))
        
        # 2. Thumb
        curr_thumb = self.calculate_angle(
            self.get_vector(lm[1], lm[2]), self.get_vector(lm[2], lm[3]))
        
        # 3. Trigger
        curr_trigger = self.calculate_angle(
            self.get_vector(lm[5], lm[6]), self.get_vector(lm[6], lm[7]))

        # 점수 계산 (오차 기반, 100점 만점)
        diff_v = abs(curr_v - self.gt_data['v_shape'])
        diff_thumb = abs(curr_thumb - self.gt_data['thumb'])
        diff_trigger = abs(curr_trigger - self.gt_data['trigger'])

        # 간단한 감점 로직 (가중치는 조절 가능)
        total_penalty = (diff_v * 1.5) + (diff_thumb * 1.0) + (diff_trigger * 1.0)
        score = max(0, 100 - int(total_penalty))

        # 피드백 생성
        feedbacks = []
        if diff_v > THRESHOLD:
            msg = "Bad V-Shape" if curr_v > self.gt_data['v_shape'] else "Widen V-Shape"
            feedbacks.append(f"[V] {msg}")
        if diff_thumb > THRESHOLD:
            msg = "Bend Thumb" if curr_thumb < self.gt_data['thumb'] else "Straighten Thumb"
            feedbacks.append(f"[T] {msg}")
        if diff_trigger > THRESHOLD:
            msg = "Pull Trigger" if curr_trigger > self.gt_data['trigger'] else "Relax Index"
            feedbacks.append(f"[I] {msg}")

        if not feedbacks and score > 85:
            feedbacks.append("Perfect Grip!")

        return score, feedbacks, (curr_v, curr_thumb, curr_trigger)

    def run(self):
        cap = cv2.VideoCapture(CAM_ID)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            # 이미지 전처리
            image = cv2.flip(image, 1) # 거울 모드
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            # 화면에 출력할 정보 초기화
            txt_y = 50

            if results.multi_hand_landmarks:
                for hand_landmarks, world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    # 랜드마크 그리기
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 3D 좌표(world_landmarks)로 평가
                    score, feedbacks, angles = self.evaluate(world_landmarks)

                    # 점수 표시 (색상: 점수 높으면 초록, 낮으면 빨강)
                    color = (0, 255, 0) if score > 80 else (0, 0, 255)
                    cv2.putText(image, f"Score: {score}", (30, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)

                    # 세부 각도 디버깅용 표시 (작게)
                    debug_str = f"V:{int(angles[0])}/{int(self.gt_data['v_shape'])}  T:{int(angles[1])}/{int(self.gt_data['thumb'])}  I:{int(angles[2])}/{int(self.gt_data['trigger'])}"
                    cv2.putText(image, debug_str, (30, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                    # 피드백 메시지 출력
                    for i, msg in enumerate(feedbacks):
                        cv2.putText(image, msg, (30, 140 + (i * 40)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(image, "Show your Hand", (30, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 1)

            cv2.imshow('Badminton Grip Coach', image)
            if cv2.waitKey(5) & 0xFF == 27: # ESC 키로 종료
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = RealTimeCoach(GT_FILE)
    coach.run()