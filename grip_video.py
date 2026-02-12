import os
import cv2
import mediapipe as mp
import time

def main():
    # 입력 영상 경로 (Grip 폴더 기준)
    VIDEO_PATH = os.path.join("grip_data", "코치_그립1.mp4")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Hands 초기화 (영상용: static_image_mode=False)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    prev_time = 0
    paused = False
    print(f"Hands video demo started: {VIDEO_PATH} (space: pause/resume, s: step, q: quit, r: restart)")

    while cap.isOpened():
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # 영상 끝 -> 종료
                break
        else:
            # 일시정지 상태에서는 이전 frame을 그대로 사용
            frame = frame

        # 영상은 보통 좌우반전(거울) 하지 않는 게 자연스럽습니다.
        # 필요하면 아래 주석 해제:
        # frame = cv2.flip(frame, 1)

        # BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Hands 추론
        results = hands.process(image_rgb)

        # 다시 BGR (그리기용)
        image_rgb.flags.writeable = True

        # 손 랜드마크 그리기
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # (선택) 4번/8번 강조 표시
                lm4 = hand_landmarks.landmark[4]
                lm8 = hand_landmarks.landmark[8]
                x4, y4 = int(lm4.x * w), int(lm4.y * h)
                x8, y8 = int(lm8.x * w), int(lm8.y * h)
                cv2.circle(frame, (x4, y4), 10, (0, 255, 255), 2)
                cv2.circle(frame, (x8, y8), 10, (0, 255, 255), 2)

        # FPS 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if paused:
            cv2.putText(
                frame,
                "PAUSED",
                (frame.shape[1]//2 - 80, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.imshow("MediaPipe Hands - Video", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
        if key == ord(" "):
            paused = not paused
        if key == ord("s") and paused:
            # 일시정지 상태에서 한 프레임씩 전진
            ok, frame = cap.read()
            if not ok:
                break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()