import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- 설정 ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\pratice_Minton\check_for_good_grip\good_grip_images"
OUTPUT_FILE = "grip_gt_range.json"

class GripFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.1
        )

    def get_vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def calculate_angle(self, center, p1, p2):
        # cos(theta) = (A . B) / (|A| * |B|)
        v1 = self.get_vector(center, p1)
        v2 = self.get_vector(center, p2)
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        dot_product = np.dot(v1, v2)
        cosine_angle = dot_product / (norm_v1 * norm_v2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine_angle))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_world_landmarks:
            print(f"⚠️ [Skip] {os.path.basename(image_path)} -> 손 인식 실패")
            return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        return {
            "v_shape": self.calculate_angle(lm[0], lm[2], lm[5]),
            "thumb": self.calculate_angle(lm[3], lm[2], lm[4]),
            "trigger_567": self.calculate_angle(lm[6], lm[5], lm[7]),
            "trigger_678": self.calculate_angle(lm[7], lm[6], lm[8])
        }

def main():
    extractor = GripFeatureExtractor()
    data_list = []
    
    if not os.path.exists(GT_FOLDER):
        print(f"❌ 오류: 폴더 없음 - {GT_FOLDER}")
        return

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
        images.extend(Path(GT_FOLDER).glob(ext.lower()))
    images = list(set(images))

    if not images:
        print("이미지가 없습니다.")
        return

    print(f"🏸 {len(images)}장 분석 시작...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        result_data = {}
        keys = ["v_shape", "thumb", "trigger_567", "trigger_678"]
        
        print("\n📊 [GT 범위 결과 (4분위수 포함)]")
        for key in keys:
            values = [d[key] for d in data_list]
            
            # --- 통계 계산 ---
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            
            # 4분위수 계산 (25%, 50%, 75%)
            q1 = np.percentile(values, 25)
            median = np.median(values) # == 50%
            q3 = np.percentile(values, 75)
            
            # IQR (Interquartile Range) - 필요시 나중에 쓰기 좋음
            iqr = q3 - q1

            result_data[key] = {
                "min": round(min_val, 2),
                "q1": round(q1, 2),        # 하위 25% 지점
                "median": round(median, 2), # 중앙값 (평균보다 이상치에 강함)
                "q3": round(q3, 2),        # 상위 25% 지점
                "max": round(max_val, 2),
                "mean": round(mean_val, 2)
            }
            
            print(f"[{key:12s}] Min: {min_val:.1f} | Q1: {q1:.1f} | Med: {median:.1f} | Q3: {q3:.1f} | Max: {max_val:.1f}")

        with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 저장 완료: {OUTPUT_FILE}")
        
    else:
        print("❌ 데이터 추출 실패")

if __name__ == "__main__":
    main()