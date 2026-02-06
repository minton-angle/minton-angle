"""
CSV 데이터로 6대 지표 계산 테스트
- gt_generate.py로 생성한 CSV 읽기
- GT 패턴 기반 Key Frame 자동 감지
- 6대 지표 계산 및 시각화
- Key Frame 이미지 추출 (분석 결과 오버레이)
"""

import pandas as pd
import numpy as np
import sys
import os
import cv2
import mediapipe as mp
import json
from badminton_metrics import BadmintonMetrics


# ========================================
# GT 패턴 학습 및 저장
# ========================================

def analyze_gt_pattern(df: pd.DataFrame, gt_frames: dict, hand: str = 'right') -> dict:
    """
    GT 프레임(26, 45, 59)의 특징 패턴 추출
    
    Args:
        df: GT CSV DataFrame
        gt_frames: {'KF1': 26, 'KF2': 45, 'KF3': 59}
        hand: 'right' 또는 'left'
    
    Returns:
        패턴 딕셔너리
    """
    
    print("\n" + "=" * 60)
    print("🎓 GT 패턴 학습 중...")
    print("=" * 60)
    
    patterns = {}
    
    for phase, frame_id in gt_frames.items():
        print(f"\n📍 {phase} (Frame {frame_id}) 특징 추출...")
        
        frame_data = df.iloc[frame_id]
        
        features = {}
        
        # 1. 손목-어깨 상대 높이
        wrist_y = frame_data[f'{hand}_wrist_y']
        shoulder_y = frame_data[f'{hand}_shoulder_y']
        elbow_y = frame_data[f'{hand}_elbow_y']
        
        features['wrist_shoulder_diff'] = float(wrist_y - shoulder_y)
        features['elbow_shoulder_diff'] = float(elbow_y - shoulder_y)
        
        # 2. 팔 각도 (어깨-팔꿈치-손목)
        shoulder = np.array([frame_data[f'{hand}_shoulder_x'], 
                           frame_data[f'{hand}_shoulder_y']])
        elbow = np.array([frame_data[f'{hand}_elbow_x'], 
                        frame_data[f'{hand}_elbow_y']])
        wrist = np.array([frame_data[f'{hand}_wrist_x'], 
                        frame_data[f'{hand}_wrist_y']])
        
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        arm_angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        features['arm_angle'] = float(arm_angle)
        
        # 3. 손목 높이 백분위수
        all_wrist_y = df[f'{hand}_wrist_y'].values
        wrist_rank = np.sum(all_wrist_y < wrist_y)
        features['wrist_percentile'] = float(wrist_rank / len(df) * 100)
        
        # 4. 팔꿈치 높이 백분위수
        all_elbow_y = df[f'{hand}_elbow_y'].values
        elbow_rank = np.sum(all_elbow_y < elbow_y)
        features['elbow_percentile'] = float(elbow_rank / len(df) * 100)
        
        # 5. 영상 내 위치 (%)
        features['position_percent'] = float(frame_id / len(df) * 100)
        
        # 6. 어깨 기울기
        left_shoulder = np.array([frame_data['left_shoulder_x'], 
                                 frame_data['left_shoulder_y']])
        right_shoulder = np.array([frame_data['right_shoulder_x'], 
                                  frame_data['right_shoulder_y']])
        
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_tilt = float(np.arctan2(shoulder_vec[1], shoulder_vec[0]) * 180 / np.pi)
        features['shoulder_tilt'] = shoulder_tilt
        
        patterns[phase] = features
        
        print(f"   ✅ 주요 특징:")
        print(f"      손목-어깨: {features['wrist_shoulder_diff']:.3f}")
        print(f"      팔꿈치-어깨: {features['elbow_shoulder_diff']:.3f}")
        print(f"      팔 각도: {features['arm_angle']:.1f}°")
        print(f"      손목 높이: 상위 {features['wrist_percentile']:.1f}%")
        print(f"      팔꿈치 높이: 상위 {features['elbow_percentile']:.1f}%")
        print(f"      영상 위치: {features['position_percent']:.1f}%")
    
    return patterns


def save_gt_pattern(patterns: dict, output_path: str):
    """GT 패턴 JSON 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)
    print(f"\n💾 GT 패턴 저장: {output_path}")


def load_gt_pattern(pattern_path: str) -> dict:
    """저장된 GT 패턴 로드"""
    if not os.path.exists(pattern_path):
        return None
    
    with open(pattern_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========================================
# GT 패턴 기반 Key Frame 감지
# ========================================

def detect_keyframes_with_pattern(df: pd.DataFrame, gt_patterns: dict, 
                                  hand: str = 'right') -> dict:
    """
    GT 패턴과 가장 유사한 프레임 찾기
    
    Args:
        df: 사용자 CSV DataFrame
        gt_patterns: GT 패턴 딕셔너리
        hand: 'right' 또는 'left'
    
    Returns:
        {'KF1': frame_id, 'KF2': frame_id, 'KF3': frame_id}
    """
    
    print("\n" + "=" * 60)
    print("🔍 GT 패턴 기반 Key Frame 감지")
    print("=" * 60)
    
    keyframes = {}
    
    for phase in ['KF1', 'KF2', 'KF3']:
        gt_feat = gt_patterns[phase]
        
        print(f"\n📍 {phase} 유사 프레임 탐색...")
        
        # 탐색 범위 (GT 위치 ±15%)
        target_pct = gt_feat['position_percent']
        start_pct = max(0, target_pct - 15)
        end_pct = min(100, target_pct + 15)
        
        start_frame = int(len(df) * start_pct / 100)
        end_frame = int(len(df) * end_pct / 100)
        
        print(f"   탐색 구간: Frame {start_frame}~{end_frame} ({start_pct:.0f}%~{end_pct:.0f}%)")
        
        best_frame = None
        best_score = float('inf')
        
        for frame_id in range(start_frame, end_frame):
            frame_data = df.iloc[frame_id]
            
            # 현재 프레임 특징
            wrist_y = frame_data[f'{hand}_wrist_y']
            shoulder_y = frame_data[f'{hand}_shoulder_y']
            elbow_y = frame_data[f'{hand}_elbow_y']
            
            wrist_sh_diff = wrist_y - shoulder_y
            elbow_sh_diff = elbow_y - shoulder_y
            
            # 팔 각도
            shoulder_xy = np.array([frame_data[f'{hand}_shoulder_x'], 
                                   frame_data[f'{hand}_shoulder_y']])
            elbow_xy = np.array([frame_data[f'{hand}_elbow_x'], 
                                frame_data[f'{hand}_elbow_y']])
            wrist_xy = np.array([frame_data[f'{hand}_wrist_x'], 
                                frame_data[f'{hand}_wrist_y']])
            
            v1 = shoulder_xy - elbow_xy
            v2 = wrist_xy - elbow_xy
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            arm_angle = np.arccos(np.clip(cos_a, -1, 1)) * 180 / np.pi
            
            # 유사도 점수 (차이의 제곱합, 가중치 적용)
            score = 0
            score += 2.0 * (wrist_sh_diff - gt_feat['wrist_shoulder_diff']) ** 2
            score += 3.0 * (elbow_sh_diff - gt_feat['elbow_shoulder_diff']) ** 2  # 가장 중요!
            score += 1.5 * ((arm_angle - gt_feat['arm_angle']) / 100) ** 2
            
            if score < best_score:
                best_score = score
                best_frame = frame_id
        
        keyframes[phase] = best_frame
        
        print(f"   ✅ 최적: Frame {best_frame} (점수: {best_score:.4f})")
        print(f"      → 위치: {best_frame/len(df)*100:.1f}% (GT: {target_pct:.1f}%)")
    
    return keyframes


def detect_keyframes_auto(df: pd.DataFrame, hand: str = 'right') -> dict:
    """
    기존 자동 감지 (GT 패턴 없을 때)
    """
    
    print("\n" + "=" * 60)
    print("3단계 Key Frame 자동 감지")
    print("=" * 60)
    
    wrist_y = df[f'{hand}_wrist_y'].values
    elbow_y = df[f'{hand}_elbow_y'].values
    shoulder_y = df[f'{hand}_shoulder_y'].values
    
    # KF1
    wrist_above_shoulder = wrist_y < shoulder_y
    kf1_candidates = np.where(wrist_above_shoulder)[0]
    kf1_frame = kf1_candidates[0] if len(kf1_candidates) > 0 else int(len(df) * 0.2)
    
    # KF2
    search_start = kf1_frame
    search_end = int(len(df) * 0.7)
    elbow_y_section = elbow_y[search_start:search_end]
    kf2_local = np.argmin(elbow_y_section)
    kf2_frame = search_start + kf2_local
    
    # KF3
    search_start = kf2_frame + 1
    wrist_y_section = wrist_y[search_start:]
    kf3_local = np.argmin(wrist_y_section)
    kf3_frame = search_start + kf3_local
    
    keyframes = {
        'KF1': kf1_frame,
        'KF2': kf2_frame,
        'KF3': kf3_frame
    }
    
    print(f"\n✅ Key Frame 감지 완료!")
    print(f"   KF1: Frame {kf1_frame} ({kf1_frame/len(df)*100:.1f}%)")
    print(f"   KF2: Frame {kf2_frame} ({kf2_frame/len(df)*100:.1f}%)")
    print(f"   KF3: Frame {kf3_frame} ({kf3_frame/len(df)*100:.1f}%)")
    
    return keyframes


# ========================================
# 시각화
# ========================================

def visualize_keyframes(df: pd.DataFrame, keyframes: dict, hand: str = 'right'):
    """Key Frame 시각화"""
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("Key Frame 시각화")
    print("=" * 60)
    
    wrist_y = df[f'{hand}_wrist_y'].values
    elbow_y = df[f'{hand}_elbow_y'].values
    shoulder_y = df[f'{hand}_shoulder_y'].values
    
    frames = np.arange(len(df))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(frames, wrist_y, label=f'{hand.capitalize()} Wrist Y', alpha=0.7)
    plt.plot(frames, elbow_y, label=f'{hand.capitalize()} Elbow Y', alpha=0.7)
    plt.plot(frames, shoulder_y, label=f'{hand.capitalize()} Shoulder Y', alpha=0.7)
    
    for phase, frame_id in keyframes.items():
        plt.axvline(x=frame_id, color='red', linestyle='--', alpha=0.5)
        plt.text(frame_id, 0.1, phase, rotation=90, verticalalignment='bottom')
    
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate (normalized)')
    plt.title('Key Frame Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    output_path = 'keyframes_visualization.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: {output_path}")
    
    plt.close()


# ========================================
# 이미지 추출 (커스터마이징 가능)
# ========================================

def extract_keyframe_images(video_path: str, df: pd.DataFrame, keyframes: dict, 
                           all_results: dict, hand: str, output_folder: str,
                           custom_prefix: str = ""):
    """
    Key Frame 이미지 추출
    
    Args:
        custom_prefix: 파일명 앞에 붙일 접두사 (예: "user1", "professional")
    """
    
    print("\n" + "=" * 60)
    print("📸 Key Frame 이미지 추출")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"\n⚠️  동영상 파일 없음: {video_path}")
        return None
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n📂 저장 폴더: {output_folder}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"\n❌ 동영상을 열 수 없습니다!")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 동영상 정보:")
    print(f"   총 프레임: {total_frames}개")
    print(f"   FPS: {fps:.2f}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
    
    saved_images = {}
    
    print(f"\n💾 Key Frame 이미지 생성 중...")
    
    phase_names = {
        'KF1': '준비 자세',
        'KF2': '백스윙',
        'KF3': '임팩트'
    }
    
    color_map = {
        '좋아요': (0, 255, 0),
        '아쉬워요': (0, 200, 255),
        '나빠요': (0, 0, 255)
    }
    
    for phase in ['KF1', 'KF2', 'KF3']:
        frame_id = keyframes[phase]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ❌ {phase}: Frame {frame_id} 추출 실패")
            continue
        
        h, w = frame.shape[:2]
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
            )
        
        # 상단 헤더
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        timestamp = frame_id / fps
        cv2.putText(frame, f"{phase}: {phase_names[phase]}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Frame {frame_id} | {timestamp:.2f}s", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # 분석 결과
        if phase in all_results:
            analysis_results = all_results[phase]
            
            y_offset = h - 40
            line_height = 35
            
            box_height = len(analysis_results) * line_height + 20
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - box_height), (550, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
            
            for metric_key, metric_data in analysis_results.items():
                name = metric_data['name']
                value = metric_data['value']
                diagnosis = metric_data['diagnosis']
                
                if 'angle' in metric_key or 'rotation' in metric_key or 'tilt' in metric_key or 'knee' in metric_key:
                    value_str = f"{value:.1f}°"
                else:
                    value_str = f"{value:.2f}"
                
                color = color_map.get(diagnosis, (255, 255, 255))
                icon = '✓' if diagnosis == '좋아요' else '!' if diagnosis == '아쉬워요' else '✗'
                
                text = f"{icon} {name}: {value_str} ({diagnosis})"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                
                y_offset -= line_height
        
        # 파일명 생성 (커스터마이징!)
        if custom_prefix:
            filename = f"{custom_prefix}_{phase}_frame_{frame_id:04d}.jpg"
        else:
            filename = f"{phase}_frame_{frame_id:04d}_analysis.jpg"
        
        image_path = os.path.join(output_folder, filename)
        cv2.imwrite(image_path, frame)
        
        saved_images[phase] = image_path
        
        print(f"   ✅ {phase}: Frame {frame_id} ({timestamp:.2f}초)")
        print(f"      → {filename}")
    
    cap.release()
    pose.close()
    
    print(f"\n✅ 총 {len(saved_images)}개 이미지 생성 완료!")
    print(f"\n📁 저장 위치: {output_folder}")
    
    return saved_images


# ========================================
# 종합 분석
# ========================================

def analyze_csv(csv_path: str, hand: str = 'right', 
                manual_keyframes: dict = None,
                video_path: str = None,
                gt_pattern_path: str = None,
                image_prefix: str = ""):
    """
    CSV 파일 종합 분석
    
    Args:
        gt_pattern_path: GT 패턴 JSON 경로 (있으면 패턴 기반 감지)
        image_prefix: 이미지 파일명 접두사
    """
    
    print("=" * 60)
    print("배드민턴 스윙 CSV 분석")
    print("=" * 60)
    
    # 1. CSV 읽기
    print(f"\n[1/5] CSV 파일 읽기...")
    print(f"   경로: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ 파일을 찾을 수 없습니다!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ CSV 로드 완료!")
    print(f"   Shape: {df.shape}")
    print(f"   프레임 수: {len(df)}개")
    
    # 2. Key Frame 감지
    print(f"\n[2/5] Key Frame 감지...")
    
    if manual_keyframes:
        keyframes = manual_keyframes
        print(f"✅ 수동 지정된 Key Frame 사용")
    elif gt_pattern_path and os.path.exists(gt_pattern_path):
        # GT 패턴 기반 감지!
        gt_patterns = load_gt_pattern(gt_pattern_path)
        print(f"✅ GT 패턴 로드: {gt_pattern_path}")
        keyframes = detect_keyframes_with_pattern(df, gt_patterns, hand)
    else:
        # 기존 자동 감지
        keyframes = detect_keyframes_auto(df, hand)
    
    # Key Frame 정보
    print(f"\n" + "=" * 60)
    print(f"📍 감지된 Key Frame 정보")
    print(f"=" * 60)
    
    for phase in ['KF1', 'KF2', 'KF3']:
        frame_id = keyframes[phase]
        frame_data = df.iloc[frame_id]
        timestamp = frame_data['timestamp']
        
        phase_names = {
            'KF1': '준비 자세',
            'KF2': '백스윙',
            'KF3': '임팩트'
        }
        
        print(f"\n🎯 {phase} ({phase_names[phase]}):")
        print(f"   프레임 번호: {frame_id}")
        print(f"   타임스탬프: {timestamp:.3f}초")
        print(f"   전체 대비: {frame_id/len(df)*100:.1f}%")
    
    # 시각화
    try:
        visualize_keyframes(df, keyframes, hand)
    except ImportError:
        print("\n⚠️  matplotlib 없음 - 그래프 건너뛰기")
    
    # 3. 각 Key Frame 분석
    print(f"\n[3/5] 6대 지표 계산...")
    
    analyzer = BadmintonMetrics()
    
    all_results = {}
    
    for phase, frame_id in keyframes.items():
        print(f"\n{'='*50}")
        print(f"📍 {phase} (Frame {frame_id}) 분석")
        print(f"{'='*50}")
        
        frame_data = df.iloc[frame_id]
        
        keypoints = {}
        for col in df.columns:
            if col not in ['frame_id', 'timestamp']:
                keypoints[col] = frame_data[col]
        
        results = analyzer.analyze_keyframe(keypoints, phase, hand)
        feedback = analyzer.generate_feedback(results, phase)
        
        print(feedback)
        
        all_results[phase] = results
    
    # 3.5. 이미지 추출
    if video_path:
        print(f"\n[3.5/5] Key Frame 이미지 추출...")
        
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_folder = os.path.join(csv_dir, f"{csv_name}_keyframes")
        
        saved_images = extract_keyframe_images(
            video_path=video_path,
            df=df,
            keyframes=keyframes,
            all_results=all_results,
            hand=hand,
            output_folder=output_folder,
            custom_prefix=image_prefix  # 커스텀 접두사!
        )
        
        if saved_images:
            print(f"\n📸 추출된 Key Frame 이미지:")
            for phase, img_path in saved_images.items():
                print(f"   {phase}: {img_path}")
    
    # 4. 종합 평가
    print(f"\n[4/5] 종합 평가...")
    
    total_good = 0
    total_ok = 0
    total_bad = 0
    total_metrics = 0
    
    for phase, results in all_results.items():
        for metric_key, metric_data in results.items():
            total_metrics += 1
            diagnosis = metric_data['diagnosis']
            
            if diagnosis == '좋아요':
                total_good += 1
            elif diagnosis == '아쉬워요':
                total_ok += 1
            else:
                total_bad += 1
    
    print(f"\n" + "=" * 60)
    print(f"📊 종합 평가")
    print(f"=" * 60)
    print(f"   총 지표: {total_metrics}개")
    print(f"   ✅ 좋아요: {total_good}개 ({total_good/total_metrics*100:.1f}%)")
    print(f"   ⚠️  아쉬워요: {total_ok}개 ({total_ok/total_metrics*100:.1f}%)")
    print(f"   ❌ 나빠요: {total_bad}개 ({total_bad/total_metrics*100:.1f}%)")
    
    score = (total_good * 10 + total_ok * 5) / total_metrics * 10
    
    print(f"\n   🎯 종합 점수: {score:.1f}/100")
    
    if score >= 80:
        grade = "S (우수)"
    elif score >= 60:
        grade = "A (양호)"
    elif score >= 40:
        grade = "B (보통)"
    else:
        grade = "C (개선 필요)"
    
    print(f"   🏆 등급: {grade}")
    
    print(f"\n" + "=" * 60)
    print(f"분석 완료!")
    print(f"=" * 60)


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("배드민턴 스윙 CSV 분석 도구")
    print("=" * 60)
    
    # 모드 선택
    print("\n💡 실행 모드 선택:")
    print("   1. GT 패턴 학습 (최초 1회)")
    print("   2. 사용자 동영상 분석")
    
    mode = input("\n선택 (1/2): ").strip()
    
    if mode == '1':
        # ===== GT 패턴 학습 =====
        print("\n" + "=" * 60)
        print("🎓 GT 패턴 학습 모드")
        print("=" * 60)
        
        gt_csv = input("\nGT CSV 경로: ").strip().strip('"').strip("'")
        
        if not os.path.exists(gt_csv):
            print("\n❌ 파일을 찾을 수 없습니다!")
            sys.exit()
        
        df = pd.read_csv(gt_csv)
        
        # GT 프레임 (26, 45, 59)
        gt_frames = {
            'KF1': 26,
            'KF2': 45,
            'KF3': 59
        }
        
        print(f"\n✅ GT Key Frames: KF1={gt_frames['KF1']}, KF2={gt_frames['KF2']}, KF3={gt_frames['KF3']}")
        
        patterns = analyze_gt_pattern(df, gt_frames, hand='right')
        
        # 저장 경로
        csv_dir = os.path.dirname(gt_csv)
        pattern_file = os.path.join(csv_dir, 'gt_pattern.json')
        
        save_gt_pattern(patterns, pattern_file)
        
        print(f"\n✅ GT 패턴 학습 완료!")
        print(f"   → 이제 사용자 동영상 분석 시 이 패턴을 사용합니다.")
        print(f"   → 패턴 파일: {pattern_file}")
    
    else:
        # ===== 사용자 동영상 분석 =====
        print("\n" + "=" * 60)
        print("👤 사용자 동영상 분석 모드")
        print("=" * 60)
        
        csv_path = input("\nCSV 경로: ").strip().strip('"').strip("'")
        
        if not csv_path:
            print("\n❌ 경로를 입력하지 않았습니다!")
            sys.exit()
        
        # 손잡이
        print("\n💡 손잡이 선택:")
        print("   1. 오른손잡이 (기본)")
        print("   2. 왼손잡이")
        
        hand_choice = input("\n선택 (1/2): ").strip()
        hand = 'left' if hand_choice == '2' else 'right'
        
        print(f"\n✅ {hand.upper()} 손잡이로 분석합니다.")
        
        # 동영상
        print("\n💡 동영상 파일 경로 (선택사항):")
        print("   Key Frame 이미지를 추출하려면 동영상 파일이 필요합니다.")
        
        video_path = input("\n동영상 경로 (Enter=건너뛰기): ").strip().strip('"').strip("'")
        
        if video_path and not os.path.exists(video_path):
            print(f"\n⚠️  동영상 파일을 찾을 수 없습니다. 건너뜁니다.")
            video_path = None
        elif video_path:
            print(f"\n✅ 동영상 파일 확인!")
        else:
            print(f"\n⏭️  동영상 없이 진행")
            video_path = None
        
        # 이미지 파일명
        image_prefix = ""
        if video_path:
            print("\n💡 이미지 파일명 접두사 (선택사항):")
            print("   예: 'user1' 입력 → 'user1_KF1_frame_0026.jpg'")
            
            image_prefix = input("\n접두사 (Enter=기본): ").strip()
            
            if image_prefix:
                print(f"\n✅ 파일명: '{image_prefix}_KF1_frame_XXXX.jpg' 형식")
            else:
                print(f"\n✅ 파일명: 기본 형식 사용")
        
        # GT 패턴
        csv_dir = os.path.dirname(csv_path)
        gt_pattern_path = os.path.join(csv_dir, 'gt_pattern.json')
        
        if not os.path.exists(gt_pattern_path):
            # 상위 폴더에서도 찾아보기
            parent_dir = os.path.dirname(csv_dir)
            gt_pattern_path = os.path.join(parent_dir, 'standard', 'gt_pattern.json')
        
        # Key Frame 지정
        print("\n💡 Key Frame 지정 방법:")
        print("   1. GT 패턴 기반 자동 감지 (권장)")
        print("   2. 기존 자동 감지")
        print("   3. 수동 입력")
        
        kf_choice = input("\n선택 (1/2/3): ").strip()
        
        manual_keyframes = None
        use_gt_pattern = (kf_choice == '1')
        
        if kf_choice == '3':
            print("\n💡 각 Key Frame의 프레임 번호를 입력하세요:")
            
            try:
                kf1 = int(input("   KF1 (준비자세): ").strip())
                kf2 = int(input("   KF2 (백스윙): ").strip())
                kf3 = int(input("   KF3 (임팩트): ").strip())
                
                manual_keyframes = {
                    'KF1': kf1,
                    'KF2': kf2,
                    'KF3': kf3
                }
                
                print(f"\n✅ Key Frame 수동 지정 완료!")
            
            except ValueError:
                print("\n⚠️  잘못된 입력! GT 패턴 기반 자동 감지로 진행합니다.")
                use_gt_pattern = True
        
        # 분석 실행
        analyze_csv(
            csv_path=csv_path,
            hand=hand,
            manual_keyframes=manual_keyframes,
            video_path=video_path,
            gt_pattern_path=gt_pattern_path if use_gt_pattern else None,
            image_prefix=image_prefix
        )