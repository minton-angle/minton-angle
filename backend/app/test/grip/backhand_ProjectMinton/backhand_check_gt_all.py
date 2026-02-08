import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path

# --- ⚙️ 설정 ---
# 본인의 백핸드 이미지 폴더 경로로 수정해주세요!
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"

# Pandas 출력 옵션
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
        """두 점 사이의 벡터 (x, y, z)"""
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        """
        두 벡터 사이의 각도 계산 (0 ~ 180도)
        공식: cos(θ) = (A · B) / (|A| * |B|)
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # 길이가 0인 벡터가 있으면 각도 계산 불가 (0 반환)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 내적 계산 후 코사인 값 구하기
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        
        # 부동소수점 오차로 1.0을 살짝 넘는 경우 방지 (-1 ~ 1 사이로 고정)
        cosine = np.clip(cosine, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        # 이미지 전처리
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 백핸드 핵심 지표 추출 ---

        # 1. Thumb Press (엄지 펴짐)
        # 2->3 벡터와 3->4 벡터 사이각. (0에 가까울수록 일직선)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 2. Index Support (검지 지지)
        # 5->6(MCP-PIP) 과 6->7(PIP-DIP) 사이각.
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 3. TI Gap (엄지-검지 사이 간격)
        # 손목(0) 기준, 엄지(2)와 검지(5)가 벌어진 각도
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        return {
            "파일명": os.path.basename(image_path), 
            "Thumb Press(엄지펴짐)": angle_thumb, 
            "Index Support(검지지지)": angle_index, 
            "TI Gap(간격)": angle_gap
        }

def show_all_details():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 이미지 파일 찾기
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))

    
    print(f"📂 경로: {GT_FOLDER}")
    print(f"📸 총 {len(images)}장의 백핸드 이미지를 분석합니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)
        else:
            print(f"⚠️ [Skip] 손 인식 실패: {img.name}")

    if not data_list:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    cols = ["파일명", "Thumb Press(엄지펴짐)", "Index Support(검지지지)", "TI Gap(간격)"]
    df = df[cols]
    df.index = np.arange(1, len(df) + 1)

    # 결과 출력
    print("="*85)
    print("📋 [백핸드] 이미지별 상세 각도 데이터")
    print("="*85)
    print(df)
    print("="*85)

if __name__ == "__main__":
    show_all_details()