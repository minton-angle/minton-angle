import cv2
import mediapipe as mp
import numpy as np
import time

def draw_landmark_indices(image, hand_landmarks, w, h, color=(255, 255, 0)):
    """Draw landmark indices (0~20) on the frame for debugging/visualization."""
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
        # 작은 배경 원(가독성)
        cv2.circle(image, (x, y), 10, (0, 0, 0), -1)
        cv2.putText(
            image,
            str(idx),
            (x - 6, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

def calculate_angle(a, b, c):
    """
    세 점 a, b, c 사이의 각도를 계산 (b가 중심점)
    """
    a = np.array(a) # 첫 번째 점
    b = np.array(b) # 중심 점 (예: 손목)
    c = np.array(c) # 세 번째 점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle



# 1. MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 모델 로드
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

# FPS 계산 변수
prev_time = 0

print("🏸 Grip Correction System Started... (Exit: q)")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # [핵심 변경 1] 읽어오자마자 좌우 반전(거울 모드) 적용
    # 이렇게 하면 이후에 그리는 글씨는 뒤집히지 않습니다.
    image = cv2.flip(image, 1)

    # 3. 전처리 (BGR -> RGB)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 4. 추론 (Inference)
    results = hands.process(image)

    # 5. 후처리 (RGB -> BGR)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 화면 크기
    h, w, c = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # (1) 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            # (1-1) 랜드마크 인덱스(0~20) 표시
            draw_landmark_indices(image, hand_landmarks, w, h)

            # (2-0) 핵심 포인트(4, 8) 먼저 체크
            lm4 = hand_landmarks.landmark[4]   # thumb tip
            lm8 = hand_landmarks.landmark[8]   # index tip

            def in_frame(lm, margin=0.02):
                return (margin <= lm.x <= 1.0 - margin) and (margin <= lm.y <= 1.0 - margin)

            has_4 = in_frame(lm4)
            has_8 = in_frame(lm8)

            # (4, 8)번 강조 표시
            if has_4:
                x4, y4 = int(lm4.x * w), int(lm4.y * h)
                cv2.circle(image, (x4, y4), 14, (0, 255, 255), 2)
            if has_8:
                x8, y8 = int(lm8.x * w), int(lm8.y * h)
                cv2.circle(image, (x8, y8), 14, (0, 255, 255), 2)

            # (2) 좌표 추출
            thumb = [hand_landmarks.landmark[4].x * w, hand_landmarks.landmark[4].y * h]
            wrist = [hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h]
            index = [hand_landmarks.landmark[8].x * w, hand_landmarks.landmark[8].y * h]

            # (3) 각도 계산
            angle = calculate_angle(thumb, wrist, index)
            
            # (4) 그립 판독 로직
            grip_status = "Check Grip"
            color = (0, 0, 255) # 빨강

            if 20 < angle < 60:
                grip_status = "Nice V-Grip!"
                color = (0, 255, 0) # 초록
            elif angle <= 20:
                grip_status = "Too Tight"
            else:
                grip_status = "Too Wide"

            # (5) 정보 시각화 (HUD) - 이제 글씨가 똑바로 나옵니다!
            cv2.putText(image, f"{int(angle)} deg", 
                        tuple(np.multiply(wrist, [1, 1]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, grip_status, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. FPS 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f"FPS: {int(fps)}", (w - 120, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # [핵심 변경 2] 마지막엔 뒤집지 않고 그대로 출력
    cv2.imshow('Badminton Grip Coach Pro', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
