import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- ⚙️ 설정 ---
# 백핸드 이미지 폴더 경로 (본인 경로 확인!)
GT_FOLDER = r"C:\Users\User\like_cool_lion\backhand_ProjectMinton\backhand_pictures\good_backhand_grip_pictures"

# --- 한글 폰트 설정 (Windows 기준: 맑은 고딕) ---
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
        """두 벡터 사이각 (0~180도)"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        
        # 내적 및 클리핑
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_world_landmarks: return None
        
        lm = results.multi_hand_world_landmarks[0].landmark
        
        # --- 🏸 백핸드 3대장 지표 ---
        # 1. Thumb Press (엄지 펴짐)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # 2. Index Support (검지 지지)
        angle_index = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
            
        # 3. TI Gap (엄지-검지 간격)
        angle_gap = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))

        return {
            "filename": os.path.basename(image_path), 
            "Thumb Press": angle_thumb, 
            "Index Support": angle_index, 
            "TI Gap": angle_gap
        }

def analyze_distribution():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 1. 이미지 파일 찾기 & 중복 제거 (set 사용)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(GT_FOLDER).glob(ext))
        images.extend(Path(GT_FOLDER).glob(ext.upper()))
    
    images = list(set(images)) # 🔥 중복 파일 제거 (12장 -> 6장 해결)
    
    print(f"📂 분석 경로: {GT_FOLDER}")
    print(f"📊 총 {len(images)}장의 정답 데이터를 분석하여 그래프를 그립니다...\n")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)

    if not data_list:
        print("❌ 데이터를 추출하지 못했습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    
    # 분석할 컬럼 (백핸드 전용)
    target_cols = ["Thumb Press", "Index Support", "TI Gap"]

    # --- 1. 통계 요약 출력 ---
    print("\n" + "="*60)
    print(" 📈 [백핸드] 데이터 통계 요약")
    print("="*60)
    print(df[target_cols].describe().round(2))
    print("="*60)

    # --- 2. 시각화 (1행 3열 그래프) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'배드민턴 백핸드 그립 GT 분포 (N={len(df)})', fontsize=16, fontweight='bold')

    # 그래프 색상
    colors = ['skyblue', 'salmon', 'lightgreen']
    
    # 각 지표별로 그래프 그리기
    for i, col in enumerate(target_cols):
        ax = axes[i]
        
        # 히스토그램 & 밀도 곡선(KDE)
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], bins=10)
        
        # 평균선 표시
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel("Angle (Degree)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # --- 3. 이상치(Outlier) 검출 ---
    print("\n🔍 이상치(Outlier) 점검:")
    print("  * Thumb Press는 30도 이상이면 '엄지가 굽었다'고 판단하여 경고합니다.")
    
    # (1) 엄지가 굽은 데이터 찾기 (백핸드에서 제일 치명적)
    bad_thumb = df[df['Thumb Press'] > 30.0]
    if not bad_thumb.empty:
        print(f"\n⚠️ [경고] 엄지가 굽어있는 사진이 {len(bad_thumb)}장 있습니다 (재촬영 권장):")
        for idx, row in bad_thumb.iterrows():
            print(f"  - {row['filename']}: {row['Thumb Press']:.1f}도")
    else:
        print("\n✅ [통과] 모든 사진의 엄지가 잘 펴져 있습니다!")

    # (2) IQR 방식의 통계적 이상치 (너무 튀는 값)
    print("\n🔍 [통계적 이상치] (IQR 기준, 너무 튀는 값):")
    for col in target_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"  [{col}] 범위 이탈 ({lower_bound:.1f} ~ {upper_bound:.1f}):")
            for idx, row in outliers.iterrows():
                print(f"    - {row['filename']}: {row[col]:.1f}도")
        else:
            print(f"  [{col}] 이상치 없음 (Clean)")

if __name__ == "__main__":
    analyze_distribution()