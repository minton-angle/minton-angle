# 엄지와 검지를 포함하는 모든 기하학적 경우의 수를 자동으로 생성하여 분석하는 통합 코드

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd
import itertools

# --- 1. 환경 설정 ---
INPUT_FOLDER = './grip_data'   
OUTPUT_BASE = './output_data'
os.makedirs(OUTPUT_BASE, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.1
)

def calculate_angle(a, b, c):
    """세 점(a, b, c)을 이용해 꼭짓점 b에서의 3D 각도를 계산"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_a, norm_c = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_a == 0 or norm_c == 0: return None
    cosang = np.dot(ba, bc) / (norm_a * norm_c)
    return round(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))), 2)

# --- 2. 모든 유효 조합 생성 함수 ---
def get_all_thumb_index_combinations():
    thumb_pts = [2, 3, 4]
    index_pts = [5, 6, 7, 8]
    all_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    valid_combos = []
    # 9개 점 중 3개를 뽑는 순열 (중간점이 꼭짓점)
    for p1, p2, p3 in itertools.permutations(all_pts, 3):
        # 중복 제거: (p1, p2, p3)와 (p3, p2, p1)은 같은 각도임
        if p1 > p3: continue 
        
        points = [p1, p2, p3]
        # 조건: 엄지 포인트 중 하나 이상 AND 검지 포인트 중 하나 이상 포함
        has_thumb = any(p in thumb_pts for p in points)
        has_index = any(p in index_pts for p in points)
        
        if has_thumb and has_index:
            valid_combos.append((p1, p2, p3))
    return valid_combos

combos_to_test = get_all_thumb_index_combinations()
print(f"📊 생성된 총 조합 수: {len(combos_to_test)}개")

# --- 3. 데이터 추출 및 전수 조사 ---
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.webp'))]
data_records = []

print(f"🚀 총 {len(image_files)}장의 이미지 분석 시작...")

for file_name in image_files:
    img_path = os.path.join(INPUT_FOLDER, file_name)
    image = cv2.imread(img_path)
    if image is None: continue
    
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_world_landmarks:
        continue

    lm = results.multi_hand_world_landmarks[0].landmark
    c = {i: [lm[i].x, lm[i].y, lm[i].z] for i in range(21)}

    features = {"file": file_name}
    for p1, p2, p3 in combos_to_test:
        angle = calculate_angle(c[p1], c[p2], c[p3])
        if angle is not None:
            features[f"A_{p2}_{p1}_{p3}"] = angle

    # 모든 조합이 성공적으로 계산된 경우만 추가
    if len(features) > len(combos_to_test) * 0.9: # 90% 이상 성공 시
        data_records.append(features)
        print(f"✅ {file_name} 분석 완료")

# --- 4. 통계 분석 및 최적 조합 선별 ---
df = pd.DataFrame(data_records)

if not df.empty:
    # 파일 컬럼 제외하고 표준편차 계산
    analysis_df = df.drop(columns=['file'])
    stats = analysis_df.std().sort_values()
    
    # 상위 3개 추출
    top_3_features = stats.head(3).index.tolist()

    guideline = {
        "version": "3.0_full_scan",
        "top_stable_combinations": top_3_features,
        "features": {}
    }

    print("\n" + "="*60)
    print("🏆 [전수조사 결과] 엄지-검지 관여 안정성 TOP 3")
    print("="*60)

    for i, feat in enumerate(top_3_features, 1):
        m = round(float(analysis_df[feat].mean()), 2)
        s = round(float(analysis_df[feat].std()), 2)
        guideline["features"][feat] = {
            "mean": m,
            "std": s,
            "min_limit": round(m - (1.5 * s), 2),
            "max_limit": round(m + (1.5 * s), 2)
        }
        print(f"{i}위: {feat:<12} | 평균: {m:>6.2f}° | 표준편차: {s:>5.2f}°")

    with open(os.path.join(OUTPUT_BASE, 'stable_guideline_full.json'), 'w') as f:
        json.dump(guideline, f, indent=4)

    print("="*60)
    print(f"✅ 전수조사 기반 가이드라인 저장 완료.")
else:
    print("\n❌ 데이터가 부족하여 통계 분석을 진행할 수 없습니다.")