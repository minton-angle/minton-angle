import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- ⚙️ 설정 ---
# 포핸드 정답 이미지가 있는 폴더
GT_FOLDER = r"C:\Users\User\like_cool_lion\forehand_ProjectMinton\check_for_good_grip\good_grip_images"
OUTPUT_FILE = "forehand_gt_range.json"

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
            print(f"⚠️ [Skip] {os.path.basename(image_path)} -> 손 인식 실패")
            return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 포핸드 4대장 지표 추출 ---
        
        # 1. V-Shape (악수)
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        # 2. Thumb Flexion (엄지 굽힘)
        # 포핸드는 감싸 쥐어야 함
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 3. Trigger (검지 방아쇠)
        angle_trigger = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 4. Grip Cross (교차각) - 🔥 핵심 판별
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        return {
            "v_shape": angle_v,
            "thumb_flex": angle_thumb,
            "trigger": angle_trigger, # Trigger 567만 사용 (PIP)
            "grip_cross": angle_cross
        }

def main():
    extractor = GripFeatureExtractor()
    data_list = []
    
    if not os.path.exists(GT_FOLDER):
        print(f"❌ 오류: 폴더 없음 - {GT_FOLDER}")
        return

    # 이미지 파일 수집 (중복 제거 포함)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    images = list(set(images))

    if not images:
        print("이미지가 없습니다.")
        return

    print(f"🏸 총 {len(images)}장의 포핸드 GT 이미지를 분석합니다...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        result_data = {}
        # 포핸드 4대장 키값
        keys = ["v_shape", "thumb_flex", "trigger", "grip_cross"]
        
        print("\n📊 [포핸드 GT 범위 결과 (JSON 생성용)]")
        print("-" * 80)
        
        for key in keys:
            values = [d[key] for d in data_list]
            
            # --- 통계 계산 ---
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            
            # 4분위수 계산
            q1 = np.percentile(values, 25)
            median = np.median(values) 
            q3 = np.percentile(values, 75)
            
            # JSON 데이터 구조
            result_data[key] = {
                "min": round(min_val, 2),
                "q1": round(q1, 2),         # 하위 25%
                "median": round(median, 2), # 중앙값
                "q3": round(q3, 2),         # 상위 25%
                "max": round(max_val, 2),
                "mean": round(mean_val, 2)
            }
            
            # 보기 좋게 출력
            print(f"[{key:15s}] Min:{min_val:5.1f} | Q1:{q1:5.1f} | Med:{median:5.1f} | Q3:{q3:5.1f} | Max:{max_val:5.1f}")

        # JSON 파일 저장
        with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
            
        print("-" * 80)
        print(f"✅ 저장 완료: {OUTPUT_FILE}")
        print("   -> 포핸드 4대장 지표가 모두 포함된 기준 파일입니다.")
        
    else:
        print("❌ 데이터 추출 실패")

if __name__ == "__main__":
    main()