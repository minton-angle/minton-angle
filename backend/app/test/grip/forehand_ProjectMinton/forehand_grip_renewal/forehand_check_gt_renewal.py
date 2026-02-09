import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- ⚙️ 설정 (포핸드 이미지 폴더 경로 확인!) ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\forehand_ProjectMinton\check_for_good_grip\good_grip_images"

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
        
        # --- 포핸드 4대장 지표 ---
        
        # 1. V-Shape (악수)
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))

        # 2. Thumb Flex (엄지 굽힘)
        # 포핸드는 감싸 쥐어야 함 (약간 굽힘)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))

        # 3. Trigger (검지 방아쇠)
        angle_trigger = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))

        # 4. Grip Cross (교차각) - 🔥 핵심 판별기
        # 포핸드는 나란해야 함 (0~40도)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        return {
            "filename": os.path.basename(image_path), 
            "V-Shape": angle_v, 
            "Thumb Flex": angle_thumb, 
            "Trigger": angle_trigger,
            "Grip Cross": angle_cross
        }

def analyze_distribution():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 이미지 로드 & 중복 제거
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    images = list(set(images)) 
    
    print(f"📂 분석 경로: {GT_FOLDER}")
    print(f"📊 총 {len(images)}장의 포핸드 데이터를 분석합니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)

    if not data_list:
        print("❌ 데이터가 없습니다.")
        return

    df = pd.DataFrame(data_list)
    
    # 포핸드 4대장
    target_cols = ["V-Shape", "Thumb Flex", "Trigger", "Grip Cross"]

    # --- 1. 통계 요약 ---
    print("\n" + "="*60)
    print(" 📈 [포핸드] 데이터 통계 요약")
    print("="*60)
    print(df[target_cols].describe().round(2))
    print("="*60)

    # --- 2. 시각화 (2x2 격자) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'배드민턴 포핸드 그립 GT 분포 (N={len(df)})', fontsize=16, fontweight='bold')

    colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
    
    for i, col in enumerate(target_cols):
        row, col_idx = divmod(i, 2)
        ax = axes[row, col_idx]
        
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], bins=8)
        
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        ax.set_title(f"{col}")
        ax.set_xlabel("Angle (Degree)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # --- 3. 이상치 점검 (포핸드 기준) ---
    print("\n🔍 [포핸드 이상치 점검]")
    
    # (1) Grip Cross 점검 (40도 이하여야 함)
    bad_cross = df[df['Grip Cross'] > 40.0]
    if not bad_cross.empty:
        print(f"⚠️ [Cross] 백핸드처럼 잡은 사진 ({len(bad_cross)}장) - 교차각이 너무 큼:")
        for idx, row in bad_cross.iterrows():
            print(f"  - {row['filename']}: {row['Grip Cross']:.1f}도 (목표: < 40)")
    else:
        print("✅ [Cross] 모두 포핸드 구조(나란함)를 갖췄습니다.")

    # (2) Thumb Flex 점검 (너무 펴지면 안됨, 10도 이상 굽어야 함)
    stiff_thumb = df[df['Thumb Flex'] < 10.0]
    if not stiff_thumb.empty:
        print(f"⚠️ [Thumb] 엄지가 너무 뻣뻣한 사진 ({len(stiff_thumb)}장):")
        for idx, row in stiff_thumb.iterrows():
            print(f"  - {row['filename']}: {row['Thumb Flex']:.1f}도 (목표: > 10)")
    else:
        print("✅ [Thumb] 엄지가 자연스럽게 굽어 있습니다.")

if __name__ == "__main__":
    analyze_distribution()