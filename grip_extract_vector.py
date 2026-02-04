import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

# --- 1. 환경 설정 ---
INPUT_FOLDER = './grip_data' 
OUTPUT_BASE = './output_data'
DIRS = ['coords', 'vectors', 'unit_vectors', 'angles', 'visuals']  # outputs

for d in DIRS:
    os.makedirs(os.path.join(OUTPUT_BASE, d), exist_ok=True)

# --- 초기화: 기존 결과물 삭제 (중복 실행 시 덮어쓰기/혼동 방지) ---
for d in DIRS:
    dir_path = os.path.join(OUTPUT_BASE, d)
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        if os.path.isfile(fp):
            os.remove(fp)
print("🧹 기존 output_data 결과물 모두 삭제 완료")

# MediaPipe 초기화 (시각화 도구 포함)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.1
)
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# --- 2. 유틸리티 함수 ---
def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def draw_landmark_indices(image, hand_landmarks, w, h, color=(255, 255, 0)):
    """Draw landmark indices (0~20) on the image."""
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
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


def in_range(val, lo, hi):
    if val is None:
        return False
    return (lo <= val <= hi)


# 각도 계산 함수 (3D)
def calculate_angle(a, b, c):
    """Return angle ABC in degrees using 3D points (a, b, c)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    # protect against zero-length vectors
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return None

    cosang = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosang)))
    return ang

# --- 3. 메인 프로세스 ---
image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
]
webp_files = [f for f in image_files if f.lower().endswith('.webp')]
print(f"Found images: {len(image_files)} (webp: {len(webp_files)})")

# GT 각도 분포(가이드라인) 생성을 위한 누적
angle_records = []

for file_name in image_files:
    img_path = os.path.join(INPUT_FOLDER, file_name)
    image = cv2.imread(img_path)
    if image is None: continue
    
    # MediaPipe 처리
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # 손 감지 실패 시 건너뜀
    if not results.multi_hand_world_landmarks:
        print(f"Skipping {file_name}: No hand detected")
        continue

    # A. 데이터 추출 (3D World Landmarks)
    world_landmarks = results.multi_hand_world_landmarks[0].landmark
    coords = {i: [lm.x, lm.y, lm.z] for i, lm in enumerate(world_landmarks)}
    save_json(coords, os.path.join(OUTPUT_BASE, 'coords', f"{file_name}_coords.json"))

    # A-1. 그립 각도(4개) 계산 및 저장
    # - 엄지 IP:  ∠(2,3,4)
    # - 검지 PIP: ∠(5,6,7)
    # - 검지 DIP: ∠(6,7,8)
    # - V각(관절 기반): ∠(3,0,6)
    angles = {
        'thumb_ip_234': calculate_angle(coords[2], coords[3], coords[4]),
        'index_pip_567': calculate_angle(coords[5], coords[6], coords[7]),
        'index_dip_678': calculate_angle(coords[6], coords[7], coords[8]),
        'v_angle_306': calculate_angle(coords[3], coords[0], coords[6]),
    }

    # None(계산 불가) 제거
    angles_clean = {k: v for k, v in angles.items() if v is not None}
    save_json(angles_clean, os.path.join(OUTPUT_BASE, 'angles', f"{file_name}_angles.json"))

    # 가이드라인 생성용 누적 (4개 모두 있어야 안정적)
    if len(angles_clean) == 4:
        angle_records.append({'file': file_name, **angles_clean})

    # B. 벡터 및 단위 벡터 계산
    vectors = {}
    unit_vectors = {}
    for start_idx, end_idx in HAND_CONNECTIONS:
        v = np.array(coords[end_idx]) - np.array(coords[start_idx])
        v_unit = get_unit_vector(v)
        key = f"{start_idx}_to_{end_idx}"
        vectors[key] = v.tolist()
        unit_vectors[key] = v_unit.tolist()

    save_json(vectors, os.path.join(OUTPUT_BASE, 'vectors', f"{file_name}_vec.json"))
    save_json(unit_vectors, os.path.join(OUTPUT_BASE, 'unit_vectors', f"{file_name}_uvec.json"))

    # C. 시각화 변형 (2D Pose Overlay)
    annotated_image = image.copy()
    
    # 이미지 위에 그릴 때는 픽셀 좌표 기준인 multi_hand_landmarks를 사용함
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # 결과 이미지 저장 (2D Pose Result)
    cv2.imwrite(os.path.join(OUTPUT_BASE, 'visuals', f"{file_name}_hands.png"), annotated_image)
    print(f"Successfully processed: {file_name}")

# --- 4. GT 가이드라인(각도 범위) 생성 ---
# 이상치에 덜 민감하도록 5~95 퍼센타일을 기본 범위로 사용
if len(angle_records) > 0:
    keys = ['thumb_ip_234', 'index_pip_567', 'index_dip_678', 'v_angle_306']
    guideline = {
        'method': 'percentile',
        'percentiles': [5, 95],
        'n_samples': len(angle_records),
        'ranges': {}
    }

    for k in keys:
        vals = [r[k] for r in angle_records]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        guideline['ranges'][k] = {
            'min': lo,
            'max': hi,
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }

    save_json(guideline, os.path.join(OUTPUT_BASE, 'grip_guideline.json'))
    print(f"\n✅ GT 가이드라인 저장 완료: {os.path.join(OUTPUT_BASE, 'grip_guideline.json')}")
else:
    print("\n GT 각도 기록이 없어 가이드라인을 생성하지 못했습니다. (손 검출 실패/angles None 등)")

# --- 5. (2nd pass) HUD 포함 시각화 이미지 생성 ---
# - 랜드마크 인덱스(0~20)
# - 4개 각도 + guideline 범위 체크(OK/FAIL)

has_guideline = (len(angle_records) > 0)

for file_name in image_files:
    img_path = os.path.join(INPUT_FOLDER, file_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hud_image = image.copy()
    h, w, _ = hud_image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                hud_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # 인덱스(0~20)
            draw_landmark_indices(hud_image, hand_landmarks, w, h)

            # (선택) 4/8 강조
            lm4 = hand_landmarks.landmark[4]
            lm8 = hand_landmarks.landmark[8]
            x4, y4 = int(lm4.x * w), int(lm4.y * h)
            x8, y8 = int(lm8.x * w), int(lm8.y * h)
            cv2.circle(hud_image, (x4, y4), 14, (0, 255, 255), 2)
            cv2.circle(hud_image, (x8, y8), 14, (0, 255, 255), 2)

        # 3D world landmark 기반 각도 계산
        if results.multi_hand_world_landmarks:
            world_landmarks = results.multi_hand_world_landmarks[0].landmark
            coords = {i: [lm.x, lm.y, lm.z] for i, lm in enumerate(world_landmarks)}
            angles_now = {
                'thumb_ip_234': calculate_angle(coords[2], coords[3], coords[4]),
                'index_pip_567': calculate_angle(coords[5], coords[6], coords[7]),
                'index_dip_678': calculate_angle(coords[6], coords[7], coords[8]),
                'v_angle_306': calculate_angle(coords[3], coords[0], coords[6]),
            }
        else:
            angles_now = {
                'thumb_ip_234': None,
                'index_pip_567': None,
                'index_dip_678': None,
                'v_angle_306': None,
            }

        ok_map = {}
        ok_count = 0
        total = 4

        if has_guideline:
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

        # HUD 박스
        cv2.rectangle(hud_image, (0, 0), (470, 165), (20, 20, 20), -1)
        cv2.putText(
            hud_image,
            f"GRIP: {status} ({ok_count}/{total})",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
            cv2.LINE_AA,
        )

        y0 = 65
        dy = 22
        order = [
            ('v_angle_306', 'V(3-0-6)'),
            ('thumb_ip_234', 'TH_IP(2-3-4)'),
            ('index_pip_567', 'IN_PIP(5-6-7)'),
            ('index_dip_678', 'IN_DIP(6-7-8)'),
        ]

        for i, (k, label) in enumerate(order):
            v = angles_now.get(k)
            ok = ok_map.get(k, False)
            c = (0, 255, 0) if ok else (0, 0, 255)

            if has_guideline:
                lo = guideline['ranges'][k]['min']
                hi = guideline['ranges'][k]['max']
                txt = f"{label}: {0 if v is None else int(v)} deg  [{int(lo)}-{int(hi)}]"
            else:
                txt = f"{label}: {0 if v is None else int(v)} deg"

            cv2.putText(
                hud_image,
                txt,
                (15, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                c,
                2,
                cv2.LINE_AA,
            )

    else:
        # 손 감지 실패
        cv2.rectangle(hud_image, (0, 0), (470, 60), (20, 20, 20), -1)
        cv2.putText(
            hud_image,
            "No hand detected",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Avoid filename collisions across different extensions by including the original extension
    base, ext = os.path.splitext(file_name)
    ext_tag = ext.lower().lstrip('.')  # e.g. '.webp' -> 'webp'
    safe_base = base.replace('.', '_')
    cv2.imwrite(
        os.path.join(OUTPUT_BASE, 'visuals', f"{safe_base}__{ext_tag}_hands_hud.png"),
        hud_image
    )

print("\n--- 모든 작업 완료! output_data/visuals 폴더를 확인해 보세요. ---")