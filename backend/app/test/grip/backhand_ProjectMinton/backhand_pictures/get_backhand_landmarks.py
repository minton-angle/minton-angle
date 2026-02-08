import cv2
import mediapipe as mp
import numpy as np
import os
import json

# --- 1. 환경 설정 ---
#INPUT_FOLDER = './my_grip_images' 
INPUT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\backhand_pictures_original"
OUTPUT_BASE = './output_data'
DIRS = ['coords', 'vectors', 'unit_vectors', 'visuals']

for d in DIRS:
    os.makedirs(os.path.join(OUTPUT_BASE, d), exist_ok=True)

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

# --- 3. 메인 프로세스 ---
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

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
    cv2.imwrite(os.path.join(OUTPUT_BASE, 'visuals', f"{file_name}_pose.png"), annotated_image)
    print(f"Successfully processed: {file_name}")

print("\n--- 모든 작업 완료! output_data/visuals 폴더를 확인해 보세요. ---")
