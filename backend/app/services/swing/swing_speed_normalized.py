"""
전문가 영상 스윙 속도 통일 (80프레임 버전)

실행: python normalize_swing_speed.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 경로 설정
PROJECT_ROOT = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend")
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.swing.engine.pose_detector import PoseDetector
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ============================================
# 설정
# ============================================
VIDEO_DIR = PROJECT_ROOT / "data" / "expert_videos"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standard" / "normalized_videos"
FRAME_DIR = PROJECT_ROOT / "data" / "standard" / "frames"

TARGET_FRAMES = 80  # ✅ 60 → 80 프레임으로 변경 (더 여유있게)
OUTPUT_FPS = 30


class SwingNormalizer:
    def __init__(self):
        self.pose_detector = PoseDetector()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
    
    def detect_swing_region(self, df):
        """스윙 구간 감지 (더 넓게 잡기)"""
        if 'right_wrist_y' not in df.columns:
            return 0, len(df) - 1
        
        wrist_y = df['right_wrist_y'].values
        wrist_y_smooth = gaussian_filter1d(wrist_y, sigma=2)
        
        peak_idx = int(np.argmin(wrist_y_smooth))
        velocity = np.gradient(wrist_y_smooth)
        
        # ✅ 시작점: 더 여유있게 (준비자세 포함)
        start_idx = 0
        for i in range(peak_idx - 1, 0, -1):
            if velocity[i] < -0.002 and velocity[i-1] >= 0:  # 민감도 낮춤
                start_idx = max(0, i - 10)  # ✅ 여유 더 줌 (3 → 10)
                break
        
        # ✅ 끝점: 더 여유있게 (팔로우스루 포함)
        end_idx = len(df) - 1
        peak_y = wrist_y_smooth[peak_idx]
        start_y = wrist_y_smooth[start_idx]
        threshold = peak_y + (start_y - peak_y) * 0.9  # ✅ 0.8 → 0.9 (더 길게)
        
        for i in range(peak_idx, len(wrist_y_smooth)):
            if wrist_y_smooth[i] > threshold:
                end_idx = min(len(df) - 1, i + 10)  # ✅ 여유 더 줌 (3 → 10)
                break
        
        # ✅ 최소 프레임 보장
        min_frames = 40
        if end_idx - start_idx < min_frames:
            center = peak_idx
            start_idx = max(0, center - min_frames // 2)
            end_idx = min(len(df) - 1, center + min_frames // 2)
        
        return start_idx, end_idx
    
    def save_video_mp4(self, frames, output_path, fps, width, height):
        """MP4 저장 (여러 코덱 시도)"""
        
        codecs = [
            ('avc1', '.mp4'),
            ('H264', '.mp4'),
            ('X264', '.mp4'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
        ]
        
        for codec, ext in codecs:
            try:
                test_path = output_path.with_suffix(ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(test_path), fourcc, fps, (width, height))
                
                if out.isOpened():
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    
                    if test_path.exists() and test_path.stat().st_size > 1000:
                        print(f"        ✅ 코덱 '{codec}' 성공!")
                        return test_path
                    else:
                        test_path.unlink(missing_ok=True)
                        
            except Exception as e:
                continue
        
        return None
    
    def normalize_video(self, video_path, expert_id):
        """단일 영상 속도 정규화"""
        
        print(f"\n{'─' * 50}")
        print(f"🎬 {expert_id} 처리 중...")
        
        # 1. 포즈 추출
        print("  [1/4] 포즈 추출...")
        pose_data = self.pose_detector.extract_from_video(str(video_path))
        df = pd.DataFrame(pose_data)
        print(f"        원본: {len(df)} 프레임")
        
        # 2. 스윙 구간 감지
        print("  [2/4] 스윙 구간 감지...")
        start_idx, end_idx = self.detect_swing_region(df)
        swing_frames = end_idx - start_idx + 1
        print(f"        스윙 구간: {start_idx} ~ {end_idx} ({swing_frames} 프레임)")
        
        # 3. 원본 영상 읽기
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        swing_video_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        for i in range(swing_frames):
            ret, frame = cap.read()
            if ret:
                swing_video_frames.append(frame)
        cap.release()
        
        print(f"        추출된 프레임: {len(swing_video_frames)}개")
        
        # 4. 리샘플링
        print(f"  [3/4] 리샘플링: {len(swing_video_frames)} → {TARGET_FRAMES} 프레임...")
        
        original_indices = np.linspace(0, len(swing_video_frames) - 1, TARGET_FRAMES)
        
        resampled_frames = []
        for i, target_idx in enumerate(original_indices):
            source_idx = int(round(target_idx))
            source_idx = min(source_idx, len(swing_video_frames) - 1)
            
            frame = swing_video_frames[source_idx].copy()
            cv2.putText(frame, f"Frame: {i}/{TARGET_FRAMES}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            resampled_frames.append(frame)
        
        # 5. 영상 저장
        print("  [4/4] 영상 저장...")
        output_path = OUTPUT_DIR / f"{expert_id}_normalized.mp4"
        saved_path = self.save_video_mp4(resampled_frames, output_path, OUTPUT_FPS, width, height)
        
        if saved_path:
            print(f"        저장: {saved_path} ({saved_path.stat().st_size // 1024} KB)")
        else:
            print(f"        ❌ 영상 저장 실패 (프레임 이미지만 저장됨)")
        
        # 6. 프레임 이미지 저장
        frame_folder = FRAME_DIR / expert_id
        frame_folder.mkdir(parents=True, exist_ok=True)
        
        # 기존 프레임 삭제
        for old_file in frame_folder.glob("*.jpg"):
            old_file.unlink()
        
        for i, frame in enumerate(resampled_frames):
            frame_path = frame_folder / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        print(f"  ✅ 프레임 {TARGET_FRAMES}개 저장: {frame_folder}")
        
        return {
            'expert_id': expert_id,
            'original_frames': len(df),
            'swing_start': start_idx,
            'swing_end': end_idx,
            'swing_frames': swing_frames,
            'speed_ratio': swing_frames / TARGET_FRAMES
        }
    
    def run(self):
        """전체 실행"""
        print("=" * 60)
        print("🏸 스윙 속도 정규화 시작")
        print(f"   목표: 모든 스윙을 {TARGET_FRAMES}프레임으로 통일")
        print("=" * 60)
        
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4']:
            video_files.extend(VIDEO_DIR.glob(ext))
        
        video_files = [f for f in video_files if f.stem.startswith('expert_')]
        video_files = sorted(set(video_files))
        
        print(f"\n📁 영상 {len(video_files)}개 발견")
        
        results = []
        for video_path in video_files:
            expert_id = video_path.stem
            result = self.normalize_video(video_path, expert_id)
            results.append(result)
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 결과 요약")
        print("=" * 60)
        print(f"{'Expert':<12} {'원본':<8} {'스윙구간':<15} {'속도비율':<10}")
        print("-" * 50)
        
        for r in results:
            ratio_text = f"{r['speed_ratio']:.2f}x"
            if r['speed_ratio'] > 1:
                ratio_text += " (느려짐)"
            else:
                ratio_text += " (빨라짐)"
            
            print(f"{r['expert_id']:<12} {r['original_frames']:<8} "
                  f"{r['swing_start']:>3}-{r['swing_end']:<8} {ratio_text}")
        
        print("\n" + "=" * 60)
        print("✅ 완료!")
        print(f"   정규화 영상: {OUTPUT_DIR}")
        print(f"   프레임 이미지: {FRAME_DIR}")
        print("=" * 60)
        print("\n📌 키프레임 찾기 가이드 (80프레임 기준):")
        print("   E1 (준비자세): 약 15~25 프레임 근처")
        print("   E2 (백스윙):   약 35~45 프레임 근처")
        print("   E3 (임팩트):   약 50~60 프레임 근처")


if __name__ == "__main__":
    normalizer = SwingNormalizer()
    normalizer.run()
