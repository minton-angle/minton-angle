
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path

# --- 설정 (경로는 그대로 유지) ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\pratice_Minton\check_for_good_grip\good_grip_images"

# Pandas 출력 옵션 설정 (데이터가 많아도 생략 없이 다 보여주기)
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

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def calculate_angle(self, v1, v2):
        v1_u = self.normalize(v1)
        v2_u = self.normalize(v2)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # 1. V-Shape (0-2-5) : Wrist(0) 기준
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))
            
        # 2. Thumb Flexion (2-3-4) : Thumb MCP(3) 기준
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # 3. Trigger PIP (5-6-7) : Index PIP(6) 기준
        angle_trigger_567 = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
            
        # 4. Trigger DIP (6-7-8) : Index DIP(7) 기준
        angle_trigger_678 = self.calculate_angle(
            self.get_vector(lm[7], lm[6]), self.get_vector(lm[7], lm[8]))

        return {
            "파일명": os.path.basename(image_path), 
            "V-Shape(0-2-5)": angle_v, 
            "Thumb(2-3-4)": angle_thumb, 
            "Trigger(5-6-7)": angle_trigger_567,
            "Trigger(6-7-8)": angle_trigger_678
        }

def show_all_details():
    extractor = GripFeatureExtractor()
    data_list = []
    
    images = list(Path(GT_FOLDER).glob("*.[jJ][pP][gG]")) + \
             list(Path(GT_FOLDER).glob("*.[pP][nN][gG]")) + \
             list(Path(GT_FOLDER).glob("*.[wW][eE][bB][pP]"))
    
    print(f"📂 경로: {GT_FOLDER}")
    print(f"📸 총 {len(images)}장의 이미지를 분석합니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)
        else:
            print(f"❌ [실패] 손 인식 불가: {img.name}")

    if not data_list:
        print("데이터가 없습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    
    # 컬럼 순서 정렬 (보기 좋게)
    cols = ["파일명", "V-Shape(0-2-5)", "Thumb(2-3-4)", "Trigger(5-6-7)", "Trigger(6-7-8)"]
    df = df[cols]

    # 1. 상세 데이터 출력
    print("="*80)
    print("📋 이미지별 상세 각도 데이터")
    print("="*80)
    # 인덱스를 1부터 시작하게 조정
    df.index = np.arange(1, len(df) + 1)
    print(df)
    print("="*80)

if __name__ == "__main__":
    show_all_details()