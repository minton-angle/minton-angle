
import cv2  
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 설정 (사용자 경로 그대로 적용) ---
GT_FOLDER = r"C:\Users\User\like_cool_lion\pratice_Minton\check_for_good_grip\good_grip_images"

# --- 한글 폰트 설정 (Windows 기준) ---
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
        
        # 1. V-Shape (Wrist 0 기준, 2번과 5번 사이 각도)
        angle_v = self.calculate_angle(
            self.get_vector(lm[0], lm[2]), self.get_vector(lm[0], lm[5]))
            
        # 2. Thumb Flexion (IP 3번 기준, 2번과 4번 사이 각도 - 2-3-4)
        angle_thumb = self.calculate_angle(
            self.get_vector(lm[3], lm[2]), self.get_vector(lm[3], lm[4]))
            
        # 3. Index Trigger PIP (PIP 6번 기준, 5번과 7번 사이 각도 - 5-6-7)
        angle_trigger_567 = self.calculate_angle(
            self.get_vector(lm[6], lm[5]), self.get_vector(lm[6], lm[7]))
            
        # 4. Index Trigger DIP (DIP 7번 기준, 6번과 8번 사이 각도 - 6-7-8)
        angle_trigger_678 = self.calculate_angle(
            self.get_vector(lm[7], lm[6]), self.get_vector(lm[7], lm[8]))

        return {
            "filename": os.path.basename(image_path), 
            "v_shape": angle_v, 
            "thumb": angle_thumb, 
            "trigger_567": angle_trigger_567,
            "trigger_678": angle_trigger_678
        }

def analyze_distribution():
    extractor = GripFeatureExtractor()
    data_list = []
    
    # 이미지 파일 찾기
    images = list(Path(GT_FOLDER).glob("*.[jJ][pP][gG]")) + \
            list(Path(GT_FOLDER).glob("*.[pP][nN][gG]")) + \
            list(Path(GT_FOLDER).glob("*.[wW][eE][bB][pP]"))
    
    print(f"총 {len(images)}장의 이미지를 분석 중입니다...")
    
    for img in images:
        result = extractor.extract(str(img))
        if result:
            data_list.append(result)

    if not data_list:
        print("데이터를 추출하지 못했습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    
    # 분석할 컬럼 리스트 정의
    target_cols = ["v_shape", "thumb", "trigger_567", "trigger_678"]

    # --- 1. 통계적 요약 출력 ---
    print("\n" + "="*60)
    print("   📊 데이터 통계 요약 (Descriptive Statistics)")
    print("="*60)
    print(df[target_cols].describe().round(2))
    print("="*60)

    # --- 2. 시각화 (Boxplot & Histogram) ---
    # 4개 지표이므로 2x2 그리드로 변경
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'배드민턴 그립 GT 데이터 분포 (N={len(df)})', fontsize=16)

    features = [
        ("v_shape", "V-Shape (0-2-5)"), 
        ("thumb", "Thumb Flexion (2-3-4)"), 
        ("trigger_567", "Trigger PIP (5-6-7)"),
        ("trigger_678", "Trigger DIP (6-7-8)")
    ]

    colors = ['skyblue', 'salmon', 'lightgreen', 'orange']

    for i, (col, title) in enumerate(features):
        row, col_idx = divmod(i, 2) # 0,0 -> 0,1 -> 1,0 -> 1,1 순서로 배치
        ax = axes[row, col_idx]

        # KDE Plot (분포 곡선)을 포함한 히스토그램
        sns.histplot(df[col], kde=True, ax=ax, color=colors[i], label='Histogram')
        
        # 평균선
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        # Boxplot (상단에 작게 오버레이 하거나 별도로 그릴 수 있지만, 여기선 분포 위주로)
        # 분포를 더 잘 보기 위해 박스플롯을 별도 축 대신 같은 축의 상단에 작게 표시하는 방식도 있지만
        # 기존 스타일대로 깔끔하게 갑니다. (히스토그램 위주)
        
        ax.set_title(f"{title} Distribution")
        ax.set_xlabel("Angle (Degree)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # --- 3. 이상치 추천 (IQR 방식) ---
    print("\n🔍 이상치(Outlier) 의심 파일:")
    
    for col, title in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"[{title}] 권장 범위 ({lower_bound:.1f} ~ {upper_bound:.1f}) 벗어남:")
            for idx, row in outliers.iterrows():
                print(f"  - {row['filename']}: {row[col]:.1f}도")
        else:
            print(f"[{title}] 이상치 없음 (Clean)")

if __name__ == "__main__":
    analyze_distribution()