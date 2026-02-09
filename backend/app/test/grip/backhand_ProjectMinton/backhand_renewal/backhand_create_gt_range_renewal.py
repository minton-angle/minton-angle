import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path

# --- ⚙️ 설정 ---
# 이미지가 있는 폴더 경로
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"
# 저장할 파일명 (이 파일이 나중에 실시간 코칭에 쓰입니다)
OUTPUT_FILE = "backhand_gt_range.json"

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
        
        # --- 🏸 백핸드 4대장 지표 추출 ---
        
        # 1. Thumb Press (엄지 펴짐)
        # 3(MCP) 중심, 180도에 가까울수록 일직선
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), 
            self.get_vector(lm[3], lm[4])
        )

        # 2. Index Support (검지 지지)
        # 6(PIP) 중심, 140~160도 아치형
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), 
            self.get_vector(lm[6], lm[7])
        )

        # 3. Grip Cross (교차각) - 구조 판별용
        # 엄지 방향 vs 검지 감는 방향 (60~90도 목표)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        # 4. TI Gap (손바닥 공간)
        # 0(Wrist) 중심, 20~40도 목표
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), 
            self.get_vector(lm[0], lm[5])
        )

        return {
            "thumb_press": angle_thumb,
            "index_support": angle_index,
            "grip_cross": angle_cross,
            "ti_gap": angle_gap
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
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
        images.extend(Path(GT_FOLDER).glob(ext.lower()))
    images = list(set(images))

    if not images:
        print("이미지가 없습니다.")
        return

    print(f"🏸 총 {len(images)}장의 백핸드 GT 이미지를 분석합니다...")
    
    for img in images:
        angles = extractor.extract(str(img))
        if angles:
            data_list.append(angles)
            print(f" - {img.name} 완료")

    if data_list:
        result_data = {}
        # 4가지 키값
        keys = ["thumb_press", "index_support", "grip_cross", "ti_gap"]
        
        print("\n📊 [백핸드 GT 범위 결과 (JSON 생성용)]")
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
        print("   -> 4대장 지표가 모두 포함된 최신 기준 파일입니다.")
        
    else:
        print("❌ 데이터 추출 실패")

if __name__ == "__main__":
    main()