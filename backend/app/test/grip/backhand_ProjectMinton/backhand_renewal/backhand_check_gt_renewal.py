import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- ⚙️ 설정 ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"

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
        
        # 1. Thumb Press (180도=일직선)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))

        # 2. Index Support (검지 아치)
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))

        # 3. Grip Cross (교차각)
        vec_thumb_dir = self.get_vector(lm[2], lm[4])
        vec_index_dir = self.get_vector(lm[5], lm[6])
        angle_cross = self.calculate_angle(vec_thumb_dir, vec_index_dir)

        # 4. TI Gap (간격)
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))

        return {
            "filename": os.path.basename(image_path), 
            "Thumb Press": angle_thumb, 
            "Index Support": angle_index, 
            "Grip Cross": angle_cross,
            "TI Gap": angle_gap
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
    print(f"📊 총 {len(images)}장의 정답 데이터를 분석합니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)

    if not data_list:
        print("❌ 데이터가 없습니다.")
        return

    df = pd.DataFrame(data_list)
    
    # 4대장 지표
    target_cols = ["Thumb Press", "Index Support", "Grip Cross", "TI Gap"]

    # --- 1. 통계 요약 ---
    print("\n" + "="*60)
    print(" 📈 [백핸드 4-Factor] 데이터 통계 요약")
    print("="*60)
    print(df[target_cols].describe().round(2))
    print("="*60)

    # --- 2. 시각화 (2x2 격자) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'배드민턴 백핸드 그립 GT 분포 (N={len(df)})', fontsize=16, fontweight='bold')

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

    # --- 3. 이상치 점검 (새로운 기준 적용) ---
    print("\n🔍 [이상치 점검 결과]")

    # (1) 엄지 점검 (180도에 가까워야 함)
    # 160도보다 작으면 엄지가 굽은 것
    bent_thumb = df[df['Thumb Press'] < 160.0]
    if not bent_thumb.empty:
        print(f"⚠️ [엄지 경고] 엄지가 굽은 사진 ({len(bent_thumb)}장):")
        for idx, row in bent_thumb.iterrows():
            print(f"  - {row['filename']}: {row['Thumb Press']:.1f}도 (목표: 170~180)")
    else:
        print("✅ [엄지 통과] 모든 사진의 엄지가 잘 펴져 있습니다.")

    # (2) 교차각 점검 (50도보다 커야 함)
    # 50도보다 작으면 포핸드나 주먹에 가까움
    bad_cross = df[df['Grip Cross'] < 50.0]
    if not bad_cross.empty:
        print(f"⚠️ [구조 경고] 백핸드 형태가 아닌 사진 ({len(bad_cross)}장):")
        for idx, row in bad_cross.iterrows():
            print(f"  - {row['filename']}: {row['Grip Cross']:.1f}도 (목표: > 60)")
    else:
        print("✅ [구조 통과] 모두 백핸드 그립 구조(T자형)를 갖췄습니다.")

if __name__ == "__main__":
    analyze_distribution()