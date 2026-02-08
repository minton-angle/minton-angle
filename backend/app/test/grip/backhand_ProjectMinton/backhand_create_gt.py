import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- ⚙️ 설정 ---
# 백핸드 이미지가 있는 폴더 (경로 확인 필수!)
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"
OUTPUT_FILE = "backhand_gt.json"   # 저장될 파일 이름

class GripFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

    def get_vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        """두 벡터 사이의 각도 (0~180도)"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        
        # 내적 계산 및 각도 변환
        dot_product = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: 
            print(f"⚠️ [손 인식 실패] {os.path.basename(image_path)}")
            return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 백핸드 핵심 지표 추출 ---

        # 1. Thumb Press (엄지 펴짐 정도)
        # 2(CMC) -> 3(MCP) 벡터와 3(MCP) -> 4(IP) 벡터 사이각
        # * 백핸드는 엄지로 눌러야 하므로 이 각도가 0에 가까워야 함 (일직선)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 2. Index Support (검지 지지 각도)
        # 5(MCP) -> 6(PIP) 벡터와 6(PIP) -> 7(DIP) 벡터 사이각
        # * 검지가 라켓 뒤를 받쳐주는 아치 형태
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 3. TI Gap (엄지-검지 간격)
        # 손목(0) 기준, 엄지(2)와 검지(5)가 벌어진 각도
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        return {
            "thumb_press": angle_thumb, 
            "index_support": angle_index, 
            "ti_gap": angle_gap
        }

def main():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 폴더 확인
    if not os.path.exists(GT_FOLDER):
        print(f"❌ 폴더가 없습니다: {GT_FOLDER}")
        return

    # 이미지 파일 찾기 (중복 제거 포함)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    
    images = list(set(images)) # 🔥 중복 파일 제거
    
    if not images:
        print("❌ 이미지가 없습니다.")
        return

    print(f"🏸 총 {len(images)}장의 백핸드 이미지를 분석하여 평균값을 계산합니다...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        # 평균값 계산 (Ground Truth)
        avg_data = {
            "thumb_press": np.mean([d["thumb_press"] for d in data_list]),
            "index_support": np.mean([d["index_support"] for d in data_list]),
            "ti_gap": np.mean([d["ti_gap"] for d in data_list])
        }
        
        # JSON 저장
        with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
            json.dump(avg_data, f, indent=4)
            
        print("\n" + "="*50)
        print(f"✅ [성공] 백핸드 기준 데이터 저장 완료: {OUTPUT_FILE}")
        print("="*50)
        print(f"🎯 [평균 기준값]")
        print(f" 1. Thumb Press (엄지 펴짐) : {avg_data['thumb_press']:.1f}도 (낮을수록 좋음)")
        print(f" 2. Index Support (검지 지지): {avg_data['index_support']:.1f}도")
        print(f" 3. TI Gap (엄지-검지 간격) : {avg_data['ti_gap']:.1f}도")
        print("="*50)
    else:
        print("❌ 유효한 데이터를 추출하지 못했습니다.")

if __name__ == "__main__":
    main()