import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json

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


def load_guideline(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def calculate_angle_3d(a, b, c):
    """Return angle ABC in degrees using 3D points (a, b, c)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return None

    cosang = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def in_range(val, lo, hi):
    if val is None:
        return False
    return (lo <= val <= hi)



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

GUIDELINE_PATH = os.path.join('output_data', 'grip_guideline.json')
guideline = load_guideline(GUIDELINE_PATH)

if guideline is None:
    print(f"⚠️ Guideline not found: {GUIDELINE_PATH} (run grip_extract_vector.py first)")
else:
    print(f"✅ Guideline loaded: {GUIDELINE_PATH} (n_samples={guideline.get('n_samples')})")

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

            # (2) 3D 좌표 추출 (가능하면 world landmark 사용)
            # results.multi_hand_world_landmarks가 있으면 그걸 우선 사용
            if results.multi_hand_world_landmarks:
                world_lms = results.multi_hand_world_landmarks[0].landmark
                coords3d = {i: [lm.x, lm.y, lm.z] for i, lm in enumerate(world_lms)}

                a_thumb_ip = calculate_angle_3d(coords3d[2], coords3d[3], coords3d[4])
                a_index_pip = calculate_angle_3d(coords3d[5], coords3d[6], coords3d[7])
                a_index_dip = calculate_angle_3d(coords3d[6], coords3d[7], coords3d[8])
                a_v = calculate_angle_3d(coords3d[3], coords3d[0], coords3d[6])
            else:
                # fallback: 2D pixel 기반 (정확도는 3D보다 떨어질 수 있음)
                p = {i: [hand_landmarks.landmark[i].x * w, hand_landmarks.landmark[i].y * h] for i in range(21)}
                a_thumb_ip = calculate_angle(p[2], p[3], p[4])
                a_index_pip = calculate_angle(p[5], p[6], p[7])
                a_index_dip = calculate_angle(p[6], p[7], p[8])
                a_v = calculate_angle(p[3], p[0], p[6])

            angles_now = {
                'thumb_ip_234': a_thumb_ip,
                'index_pip_567': a_index_pip,
                'index_dip_678': a_index_dip,
                'v_angle_306': a_v,
            }

            # (3) 가이드라인 범위 체크
            ok_map = {}
            ok_count = 0
            total = 4

            if guideline and 'ranges' in guideline:
                for k, v in angles_now.items():
                    lo = guideline['ranges'][k]['min']
                    hi = guideline['ranges'][k]['max']
                    ok = in_range(v, lo, hi)
                    ok_map[k] = ok
                    ok_count += 1 if ok else 0

                passed = (ok_count == total)
                status = 'OK' if passed else 'FAIL'
                status_color = (0, 255, 0) if passed else (0, 0, 255)
            else:
                status = 'NO_GUIDELINE'
                status_color = (0, 0, 255)
                for k in angles_now.keys():
                    ok_map[k] = False

            # (4) HUD 표시
            cv2.rectangle(image, (0, 0), (420, 150), (20, 20, 20), -1)
            cv2.putText(image, f"GRIP: {status} ({ok_count}/{total})", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

            # 각도/범위 표시
            y0 = 65
            dy = 20
            order = [
                ('v_angle_306', 'V(3-0-6)'),
                ('thumb_ip_234', 'TH_IP(2-3-4)'),
                ('index_pip_567', 'IN_PIP(5-6-7)'),
                ('index_dip_678', 'IN_DIP(6-7-8)'),
            ]

            for i, (k, label) in enumerate(order):
                v = angles_now[k]
                ok = ok_map.get(k, False)
                c = (0, 255, 0) if ok else (0, 0, 255)

                if guideline and 'ranges' in guideline:
                    lo = guideline['ranges'][k]['min']
                    hi = guideline['ranges'][k]['max']
                    txt = f"{label}: {0 if v is None else int(v)} deg  [{int(lo)}-{int(hi)}]"
                else:
                    txt = f"{label}: {0 if v is None else int(v)} deg"

                cv2.putText(image, txt, (15, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)

    # 6. FPS 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f"FPS: {int(fps)}", (w - 120, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # [핵심 변경 2] 마지막엔 뒤집지 않고 그대로 출력
    cv2.imshow('Badminton Grip Coach - Guideline Check', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
