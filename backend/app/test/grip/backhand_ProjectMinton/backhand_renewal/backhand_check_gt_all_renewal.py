import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path

# --- ⚙️ 설정 ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

class GripFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

    def get_vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 1. Thumb Press (엄지 펴짐: Power) ---
        # 180도에 가까울수록 좋습니다 (일직선)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # --- 🏸 2. Index Support (검지 지지: Control) ---
        # 140~160도 사이의 아치형
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # --- 🏸 3. Grip Cross (교차각: Structure) ---
        # 엄지 방향 vs 검지 감는 방향. 60~90도면 백핸드(T자형)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        # --- 🏸 4. TI Gap (손바닥 공간: Mobility) --- 🔥 부활!
        # 손목(0) 기준 엄지(2)와 검지(5) 사이 각도
        # 20~40도 (너무 좁으면 손목 잠김, 너무 넓으면 헐거움)
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        return {
            "파일명": os.path.basename(image_path), 
            "Thumb Press": angle_thumb, 
            "Index Support": angle_index, 
            "Grip Cross": angle_cross,
            "TI Gap": angle_gap
        }

def show_all_details():
    extractor = GripFeatureExtractor()
    data_list = []
    
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    
    images = list(set(images)) 
    
    print(f"📂 경로: {GT_FOLDER}")
    print(f"📸 총 {len(images)}장의 이미지를 분석합니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)
        else:
            print(f"⚠️ [Skip] 인식 실패: {img.name}")

    if not data_list:
        print("❌ 데이터가 없습니다.")
        return

    df = pd.DataFrame(data_list)
    # 컬럼 순서 정리
    cols = ["파일명", "Thumb Press", "Index Support", "Grip Cross", "TI Gap"]
    df = df[cols]
    df.index = np.arange(1, len(df) + 1)

    print("="*100)
    print("📋 [백핸드 4대장 지표] 상세 분석 결과")
    print(" 1. Thumb Press  : 170~180도 (엄지 펴짐)")
    print(" 2. Index Support: 140~160도 (검지 아치)")
    print(" 3. Grip Cross   : 60~90도   (백핸드 구조)")
    print(" 4. TI Gap       : 20~40도   (손목 공간)")
    print("="*100)
    print(df)
    print("="*100)

if __name__ == "__main__":
    show_all_details()