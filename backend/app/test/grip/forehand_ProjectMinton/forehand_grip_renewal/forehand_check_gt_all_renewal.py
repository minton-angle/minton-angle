import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path

# --- ⚙️ 설정 (포핸드 이미지 폴더 경로로 수정!) ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\forehand_ProjectMinton\check_for_good_grip\good_grip_images"

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
        """두 점 사이의 벡터 (p1 -> p2)"""
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, v1, v2):
        """두 벡터 사이의 각도 계산 (0 ~ 180도)"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 포핸드 핵심 지표 추출 ---

        # 1. V-Shape (악수하는 모양)
        # 손목(0) 기준 엄지(2)와 검지(5) 벌림 각도
        # * 포핸드는 악수하듯 적당히 벌어져야 함 (20~50도)
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        # 2. Thumb Flexion (엄지 굽힘)
        # 2(CMC) -> 3(MCP) -> 4(IP)
        # * 포핸드는 엄지가 라켓을 감싸야 하므로 약간 굽어야 함 (10~40도 굽힘)
        # * (주의: 180도 기준이 아니라 벡터 사이각이므로, 0에 가까울수록 펴진 것, 클수록 굽은 것)
        # * 여기서는 '얼마나 꺾였나'를 봅니다.
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 3. Trigger (검지 방아쇠)
        # 5(MCP) -> 6(PIP) -> 7(DIP)
        # * 검지가 방아쇠 당기듯 걸려 있어야 함
        angle_trigger = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 4. Grip Cross (교차각) - 🔥 백핸드와 구분하는 핵심!
        # 엄지 방향(2->4) vs 검지 감는 방향(5->6)
        # * 포핸드: 두 손가락이 나란히 감김 -> 각도가 작아야 함 (0~40도)
        # * 백핸드: 서로 엇갈림 -> 각도가 큼 (50~90도)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        return {
            "파일명": os.path.basename(image_path), 
            "V-Shape": angle_v, 
            "Thumb Flex": angle_thumb, 
            "Trigger": angle_trigger,
            "Grip Cross(교차각)": angle_cross
        }

def show_all_details():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 이미지 파일 찾기 (중복 제거)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    
    images = list(set(images)) 
    
    print(f"📂 경로: {GT_FOLDER}")
    print(f"📸 총 {len(images)}장의 포핸드 이미지를 분석합니다...\n")
    
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
    cols = ["파일명", "V-Shape", "Thumb Flex", "Trigger", "Grip Cross(교차각)"]
    df = df[cols]
    
    # 인덱스 조정
    df.index = np.arange(1, len(df) + 1)

    print("="*100)
    print("📋 [포핸드 그립] 이미지별 상세 분석 결과")
    print("="*100)
    print(df)
    print("="*100)
    print("💡 [포핸드 판별 기준]")
    print(" 1. Grip Cross : 40도 이하 (나란해야 함! 높으면 백핸드임)")
    print(" 2. Thumb Flex : 10도 이상 (약간 굽어야 함, 너무 펴지면 안됨)")
    print(" 3. V-Shape    : 적당한 악수 간격")
    print("="*100)

if __name__ == "__main__":
    show_all_details()