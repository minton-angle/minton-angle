import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- ⚙️ 설정 ---
# 포핸드 정답 이미지가 있는 폴더
GT_FOLDER = r"C:\Users\User\like_cool_lion\forehand_ProjectMinton\check_for_good_grip\good_grip_images"
# 저장할 정답 파일명
OUTPUT_FILE = "forehand_gt.json"

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
        
        # --- 🏸 포핸드 4대장 지표 추출 ---

        # 1. V-Shape (악수하는 모양)
        # 손목(0) 기준 엄지(2)와 검지(5) 사이 각도
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        # 2. Thumb Flexion (엄지 굽힘)
        # 포핸드는 엄지가 라켓을 감싸야 하므로 약간 굽어야 함 (10~40도)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 3. Trigger (검지 방아쇠)
        # 검지 관절이 잘 걸려 있는지 확인 (PIP 기준)
        angle_trigger = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 4. Grip Cross (교차각) - 🔥 포핸드/백핸드 구분 핵심
        # 엄지 방향 vs 검지 감는 방향
        # * 포핸드는 이 각도가 작아야 함 (0~40도, 나란함)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        return {
            "v_shape": angle_v, 
            "thumb_flex": angle_thumb, 
            "trigger": angle_trigger,
            "grip_cross": angle_cross
        }

def main():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 폴더 확인
    if not os.path.exists(GT_FOLDER):
        print(f"❌ 폴더가 없습니다: {GT_FOLDER}")
        return

    # 이미지 파일 수집 (중복 제거)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    images = list(set(images))
    
    if not images:
        print("❌ 이미지가 없습니다.")
        return

    print(f"🏸 총 {len(images)}장의 포핸드 이미지를 분석합니다...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        # 평균값 계산 (Ground Truth Mean)
        avg_data = {
            "v_shape": np.mean([d["v_shape"] for d in data_list]),
            "thumb_flex": np.mean([d["thumb_flex"] for d in data_list]),
            "trigger": np.mean([d["trigger"] for d in data_list]),
            "grip_cross": np.mean([d["grip_cross"] for d in data_list])
        }
        
        # JSON 저장
        with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
            json.dump(avg_data, f, indent=4)
            
        print("\n" + "="*60)
        print(f"✅ [성공] 포핸드 기준 데이터 저장 완료: {OUTPUT_FILE}")
        print("="*60)
        print(f"🎯 [포핸드 4대장 평균값]")
        print(f" 1. V-Shape    : {avg_data['v_shape']:.1f}도 (악수 간격)")
        print(f" 2. Thumb Flex : {avg_data['thumb_flex']:.1f}도 (약간 굽힘)")
        print(f" 3. Trigger    : {avg_data['trigger']:.1f}도 (검지 걸기)")
        print(f" 4. Grip Cross : {avg_data['grip_cross']:.1f}도 (목표: < 40 나란함!)")
        print("="*60)
    else:
        print("❌ 유효한 데이터를 추출하지 못했습니다.")

if __name__ == "__main__":
    main()