"""
수동 레이블링한 키프레임 이미지 추출

입력: keyframe_labels.csv
출력: 
  - keyframe_images/expert_1/E1_ready.jpg, E2_backswing.jpg, E3_impact.jpg
  - keyframe_comparison.png (10명 x 3개 한눈에 비교)

실행: python extract_keyframe_images.py
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_DIR = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos")
OUTPUT_DIR = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\keyframe_images")
LABELS_PATH = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\keyframe_labels.csv")


def extract_keyframes():
    """키프레임 이미지 추출"""
    
    print("=" * 60)
    print("📸 키프레임 이미지 추출")
    print("=" * 60)
    
    # CSV 읽기
    labels = pd.read_csv(LABELS_PATH)
    print(f"✅ 레이블 로드: {len(labels)}명\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 비교 이미지용 저장
    all_frames = {}  # expert_id -> {'E1': img, 'E2': img, 'E3': img}
    
    for _, row in labels.iterrows():
        expert_id = row['expert_id']
        e1_idx = int(row['E1_ready'])
        e2_idx = int(row['E2_backswing'])
        e3_idx = int(row['E3_impact'])
        
        print(f"🎬 {expert_id}: E1={e1_idx}, E2={e2_idx}, E3={e3_idx}")
        
        # 영상 열기
        video_path = VIDEO_DIR / f"{expert_id}.mp4"
        if not video_path.exists():
            video_path = VIDEO_DIR / f"{expert_id}.MP4"
        
        if not video_path.exists():
            print(f"   ❌ 영상 없음: {video_path}")
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        
        # 저장 폴더
        save_dir = OUTPUT_DIR / expert_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_frames[expert_id] = {}
        
        # 각 키프레임 추출
        for label, frame_idx in [('E1_ready', e1_idx), ('E2_backswing', e2_idx), ('E3_impact', e3_idx)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 프레임 정보 표시
                display = frame.copy()
                cv2.putText(display, f"{label} (F:{frame_idx})", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 저장
                save_path = save_dir / f"{label}.jpg"
                cv2.imwrite(str(save_path), display)
                
                # 비교용 저장 (라벨 없는 버전)
                all_frames[expert_id][label.split('_')[0]] = frame
                
                print(f"   ✅ {label}.jpg")
            else:
                print(f"   ❌ {label} 추출 실패")
        
        cap.release()
    
    # 비교 이미지 생성
    create_comparison_image(all_frames, labels)
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print(f"   개별 이미지: {OUTPUT_DIR}")
    print(f"   비교 이미지: {OUTPUT_DIR / 'keyframe_comparison.png'}")
    print("=" * 60)


def create_comparison_image(all_frames, labels):
    """10명 x 3개 비교 이미지 생성"""
    
    print("\n📊 비교 이미지 생성 중...")
    
    n_experts = len(all_frames)
    
    if n_experts == 0:
        print("   ❌ 추출된 프레임 없음")
        return
    
    # 그리드 설정
    fig, axes = plt.subplots(n_experts, 3, figsize=(15, 4 * n_experts))
    
    if n_experts == 1:
        axes = [axes]
    
    # 정렬 (expert_1, expert_2, ... 순서)
    sorted_experts = sorted(all_frames.keys(), key=lambda x: int(x.split('_')[1]))
    
    for i, expert_id in enumerate(sorted_experts):
        frames = all_frames[expert_id]
        
        # 레이블에서 프레임 번호 가져오기
        row = labels[labels['expert_id'] == expert_id].iloc[0]
        
        for j, (key, title, frame_col) in enumerate([
            ('E1', 'E1: 준비자세', 'E1_ready'),
            ('E2', 'E2: 백스윙', 'E2_backswing'),
            ('E3', 'E3: 임팩트', 'E3_impact')
        ]):
            ax = axes[i][j] if n_experts > 1 else axes[j]
            
            if key in frames:
                img = cv2.cvtColor(frames[key], cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                
                frame_num = int(row[frame_col])
                ax.set_title(f"{title} (F:{frame_num})", fontsize=10)
            else:
                ax.text(0.5, 0.5, '없음', ha='center', va='center')
                ax.set_facecolor('#f0f0f0')
            
            if j == 0:
                ax.set_ylabel(expert_id, fontsize=11, fontweight='bold')
            
            ax.axis('off')
    
    plt.suptitle('전문가 키프레임 비교\n(E1: 준비자세 / E2: 백스윙 / E3: 임팩트)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "keyframe_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ keyframe_comparison.png")


if __name__ == "__main__":
    extract_keyframes()