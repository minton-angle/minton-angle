import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- 설정 ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\pratice_Minton\check_for_good_grip\good_grip_images"     # 정답 이미지가 있는 폴더
OUTPUT_FILE = "grip_gt.json"   # 저장할 정답 데이터 파일

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
        if not results.multi_hand_world_landmarks: 
            print(f"⚠️ [손 인식 실패] {os.path.basename(image_path)} -> 손이 잘렸거나, 너무 흐리거나, 배경이 복잡함")
            return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # 1. V-Shape (엄지-검지 벌림) : Wrist(0) 기준
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), # Wrist -> Thumb CMC
            self.get_vector(lm[0], lm[5])  # Wrist -> Index MCP
        )

        # 2. Thumb Flexion (엄지 굽힘)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[1], lm[2]), # Thumb CMC -> MCP
            self.get_vector(lm[2], lm[3])  # Thumb MCP -> IP
        )

        # 3. Index Trigger (검지 방아쇠)
        angle_trigger = self.calculate_angle(
            self.get_vector(lm[5], lm[6]), # Index MCP -> PIP
            self.get_vector(lm[6], lm[7])  # Index PIP -> DIP
        )

        return {"v_shape": angle_v, "thumb": angle_thumb, "trigger": angle_trigger}

def main():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 폴더가 없으면 생성
    os.makedirs(GT_FOLDER, exist_ok=True)
    images = list(Path(GT_FOLDER).glob("*.[jJ][pP][gG]")) + \
         list(Path(GT_FOLDER).glob("*.[pP][nN][gG]")) + \
         list(Path(GT_FOLDER).glob("*.[wW][eE][bB][pP]"))
    
    if not images:
        print(f"[{GT_FOLDER}] 폴더에 이미지가 없습니다. 올바른 그립 사진을 넣어주세요.")
        return

    print(f"총 {len(images)}장의 GT 이미지를 분석합니다...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        # 평균값 계산
        avg_data = {
            "v_shape": np.mean([d["v_shape"] for d in data_list]),
            "thumb": np.mean([d["thumb"] for d in data_list]),
            "trigger": np.mean([d["trigger"] for d in data_list])
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(avg_data, f, indent=4)
        print(f"\n[성공] 정답 데이터 저장 완료: {OUTPUT_FILE}")
        print(f"기준 각도 -> V홈: {avg_data['v_shape']:.1f}, 엄지: {avg_data['thumb']:.1f}, 검지: {avg_data['trigger']:.1f}")
    else:
        print("[실패] 손을 인식할 수 있는 이미지가 없습니다.")

if __name__ == "__main__":
    main()