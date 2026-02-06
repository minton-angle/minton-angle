"""
3개 GT 동영상의 평균 패턴 학습
- 각 GT의 Key Frame 번호 입력
- 3개 평균 패턴 계산 및 저장
- 새로운 동영상 분석에 사용
"""

import pandas as pd
import numpy as np
import json
import os
import sys


def analyze_single_gt_pattern(csv_path: str, keyframes: dict, hand: str = 'right') -> dict:
    """
    단일 GT의 패턴 추출
    
    Args:
        csv_path: GT CSV 경로
        keyframes: {'KF1': 26, 'KF2': 45, 'KF3': 57}
        hand: 'right' 또는 'left'
    
    Returns:
        패턴 딕셔너리
    """
    
    df = pd.read_csv(csv_path)
    patterns = {}
    
    for phase, frame_id in keyframes.items():
        if frame_id >= len(df):
            print(f"   ⚠️  {phase}: Frame {frame_id}가 범위를 벗어남 (총 {len(df)}프레임)")
            continue
        
        frame_data = df.iloc[frame_id]
        features = {}
        
        # 1. 손목-어깨 상대 높이
        wrist_y = frame_data[f'{hand}_wrist_y']
        shoulder_y = frame_data[f'{hand}_shoulder_y']
        elbow_y = frame_data[f'{hand}_elbow_y']
        
        features['wrist_shoulder_diff'] = float(wrist_y - shoulder_y)
        features['elbow_shoulder_diff'] = float(elbow_y - shoulder_y)
        
        # 2. 팔 각도
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
        
        # 7. 무릎 각도 (오른쪽)
        hip = np.array([frame_data[f'{hand}_hip_x'], 
                       frame_data[f'{hand}_hip_y']])
        knee = np.array([frame_data[f'{hand}_knee_x'], 
                        frame_data[f'{hand}_knee_y']])
        ankle = np.array([frame_data[f'{hand}_ankle_x'], 
                         frame_data[f'{hand}_ankle_y']])
        
        v1_leg = hip - knee
        v2_leg = ankle - knee
        cos_knee = np.dot(v1_leg, v2_leg) / (np.linalg.norm(v1_leg) * np.linalg.norm(v2_leg) + 1e-6)
        knee_angle = np.arccos(np.clip(cos_knee, -1, 1)) * 180 / np.pi
        features['knee_angle'] = float(knee_angle)
        
        patterns[phase] = features
    
    return patterns


def calculate_average_pattern(gt_data_list: list) -> dict:
    """
    여러 GT의 평균 패턴 계산
    
    Args:
        gt_data_list: [
            {'name': 'GT1', 'csv': 'path', 'keyframes': {...}, 'pattern': {...}},
            ...
        ]
    
    Returns:
        평균 패턴 딕셔너리
    """
    
    print("\n" + "=" * 60)
    print("📊 평균 패턴 계산 중...")
    print("=" * 60)
    
    avg_pattern = {}
    
    for phase in ['KF1', 'KF2', 'KF3']:
        print(f"\n🎯 {phase} 평균 계산...")
        
        # 각 GT의 해당 phase 특징 수집
        all_features = {}
        
        for gt_data in gt_data_list:
            if phase in gt_data['pattern']:
                phase_features = gt_data['pattern'][phase]
                
                for key, value in phase_features.items():
                    if key not in all_features:
                        all_features[key] = []
                    all_features[key].append(value)
        
        # 평균 및 표준편차 계산
        avg_features = {}
        
        for key, values in all_features.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            avg_features[key] = float(avg_value)
            avg_features[f'{key}_std'] = float(std_value)
            avg_features[f'{key}_min'] = float(min_value)
            avg_features[f'{key}_max'] = float(max_value)
            
            print(f"   {key}:")
            print(f"      평균: {avg_value:.3f}")
            print(f"      표준편차: {std_value:.3f}")
            print(f"      범위: [{min_value:.3f}, {max_value:.3f}]")
        
        avg_pattern[phase] = avg_features
    
    # 메타데이터 추가
    avg_pattern['_metadata'] = {
        'num_samples': len(gt_data_list),
        'gt_names': [gt['name'] for gt in gt_data_list],
        'gt_keyframes': {gt['name']: gt['keyframes'] for gt in gt_data_list},
        'hand': 'right'
    }
    
    return avg_pattern


def save_gt_pattern(pattern: dict, output_path: str):
    """패턴을 JSON으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pattern, f, indent=2, ensure_ascii=False)
    print(f"\n💾 평균 패턴 저장: {output_path}")


def print_pattern_summary(pattern: dict):
    """패턴 요약 출력"""
    
    print("\n" + "=" * 60)
    print("📋 최종 평균 패턴 요약")
    print("=" * 60)
    
    meta = pattern['_metadata']
    
    print(f"\n📚 사용된 GT:")
    for name in meta['gt_names']:
        keyframes = meta['gt_keyframes'][name]
        print(f"   - {name}: KF1={keyframes['KF1']}, KF2={keyframes['KF2']}, KF3={keyframes['KF3']}")
    
    print(f"\n📊 평균 위치 (영상 내 %):")
    for phase in ['KF1', 'KF2', 'KF3']:
        pos = pattern[phase]['position_percent']
        pos_std = pattern[phase]['position_percent_std']
        print(f"   {phase}: {pos:.1f}% ± {pos_std:.1f}%")
    
    print(f"\n📐 주요 특징 (평균 ± 표준편차):")
    
    for phase in ['KF1', 'KF2', 'KF3']:
        phase_names = {'KF1': '준비자세', 'KF2': '백스윙', 'KF3': '임팩트'}
        print(f"\n   {phase} ({phase_names[phase]}):")
        
        feat = pattern[phase]
        
        print(f"      팔꿈치-어깨: {feat['elbow_shoulder_diff']:.3f} ± {feat['elbow_shoulder_diff_std']:.3f}")
        print(f"      팔 각도: {feat['arm_angle']:.1f}° ± {feat['arm_angle_std']:.1f}°")
        print(f"      무릎 각도: {feat['knee_angle']:.1f}° ± {feat['knee_angle_std']:.1f}°")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("3개 GT 평균 패턴 학습")
    print("=" * 60)
    
    # GT 데이터 입력
    gt_data_list = []
    
    # 미리 정의된 데이터
    predefined_gts = [
        {
            'name': 'pro1',
            'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\pro1_gt_raw.csv",
            'keyframes': {'KF1': 26, 'KF2': 45, 'KF3': 57}
        },
        {
            'name': 'GT1',
            'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT1_gt_raw.csv",
            'keyframes': {'KF1': 44, 'KF2': 56, 'KF3': 64}
        },
        {
            'name': 'GT2',
            'csv': r"C:\Users\User\Desktop\CV\FinalProj\data\GT2_gt_raw.csv",
            'keyframes': {'KF1': 39, 'KF2': 51, 'KF3': 58}
        }
    ]
    
    print("\n💡 사용할 GT 데이터:")
    print("   1. 미리 정의된 3개 GT 사용 (pro1, GT1, GT2)")
    print("   2. 수동으로 입력")
    
    choice = input("\n선택 (1/2): ").strip()
    
    if choice == '1':
        # 미리 정의된 GT 사용
        print("\n✅ 미리 정의된 GT 사용:")
        for gt in predefined_gts:
            print(f"   - {gt['name']}: {os.path.basename(gt['csv'])}")
        
        use_predefined = True
    else:
        use_predefined = False
    
    if use_predefined:
        gt_configs = predefined_gts
    else:
        # 수동 입력
        gt_configs = []
        
        print("\n" + "=" * 60)
        print("GT 데이터 입력 (3개)")
        print("=" * 60)
        
        for i in range(1, 4):
            print(f"\n[GT {i}/3]")
            
            name = input(f"   이름 (예: GT{i}): ").strip()
            if not name:
                name = f"GT{i}"
            
            csv_path = input(f"   CSV 경로: ").strip().strip('"').strip("'")
            
            if not os.path.exists(csv_path):
                print(f"   ⚠️  파일 없음 - 건너뜀")
                continue
            
            print(f"   Key Frame 번호:")
            kf1 = int(input(f"      KF1 (준비자세): ").strip())
            kf2 = int(input(f"      KF2 (백스윙): ").strip())
            kf3 = int(input(f"      KF3 (임팩트): ").strip())
            
            gt_configs.append({
                'name': name,
                'csv': csv_path,
                'keyframes': {'KF1': kf1, 'KF2': kf2, 'KF3': kf3}
            })
            
            print(f"   ✅ {name} 등록")
    
    if len(gt_configs) == 0:
        print("\n❌ 유효한 GT가 없습니다!")
        sys.exit()
    
    print(f"\n✅ 총 {len(gt_configs)}개 GT 등록됨")
    
    # 각 GT 패턴 추출
    print("\n" + "=" * 60)
    print("🎓 각 GT 패턴 추출 중...")
    print("=" * 60)
    
    for i, gt_config in enumerate(gt_configs, 1):
        print(f"\n[{i}/{len(gt_configs)}] {gt_config['name']} 분석 중...")
        
        try:
            pattern = analyze_single_gt_pattern(
                csv_path=gt_config['csv'],
                keyframes=gt_config['keyframes'],
                hand='right'
            )
            
            gt_config['pattern'] = pattern
            
            print(f"   ✅ 패턴 추출 완료")
            
            # 주요 정보 출력
            for phase in ['KF1', 'KF2', 'KF3']:
                if phase in pattern:
                    pos = pattern[phase]['position_percent']
                    print(f"      {phase}: {pos:.1f}% 위치")
        
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            continue
    
    # 유효한 GT만 필터링
    valid_gts = [gt for gt in gt_configs if 'pattern' in gt]
    
    if len(valid_gts) == 0:
        print("\n❌ 패턴 추출에 성공한 GT가 없습니다!")
        sys.exit()
    
    print(f"\n✅ {len(valid_gts)}개 GT 패턴 추출 성공")
    
    # 평균 패턴 계산
    avg_pattern = calculate_average_pattern(valid_gts)
    
    # 요약 출력
    print_pattern_summary(avg_pattern)
    
    # 저장 경로
    print("\n" + "=" * 60)
    print("💾 패턴 저장")
    print("=" * 60)
    
    # 기본 저장 경로
    first_csv_dir = os.path.dirname(gt_configs[0]['csv'])
    default_output = os.path.join(first_csv_dir, 'gt_average_pattern.json')
    
    print(f"\n기본 저장 경로: {default_output}")
    
    custom_path = input("\n다른 경로 사용? (Enter=기본): ").strip().strip('"').strip("'")
    
    output_path = custom_path if custom_path else default_output
    
    # 저장
    save_gt_pattern(avg_pattern, output_path)
    
    print("\n" + "=" * 60)
    print("🎉 평균 패턴 학습 완료!")
    print("=" * 60)
    
    print(f"\n📁 저장된 파일: {output_path}")
    print(f"\n💡 다음 단계:")
    print(f"   1. test_metrics.py 실행")
    print(f"   2. 사용자 동영상 분석 모드 선택")
    print(f"   3. Key Frame 지정: GT 패턴 기반 자동 감지")
    print(f"   → 이 평균 패턴이 자동으로 사용됩니다!")