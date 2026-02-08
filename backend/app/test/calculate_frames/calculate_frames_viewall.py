import pandas as pd
import numpy as np
import os

def calculate_badminton_averages_with_details(norm_files, kf_files, output_path):
    """
    norm_files: 4개의 정규화된 CSV 파일 경로 리스트
    kf_files: 4개의 키프레임 CSV 파일 경로 리스트
    output_path: 결과 CSV 저장 경로
    """
    
    stages = ['ready', 'backswing', 'impact']
    all_rows = [] # 모든 행(GT + Average)을 담을 리스트

    # 1. 각 단계별로 데이터 수집 및 평균 계산
    for stage in stages:
        stage_data = [] # 현재 단계의 GT1~4 데이터를 임시 저장
        
        for i, (norm_path, kf_path) in enumerate(zip(norm_files, kf_files), 1):
            norm_df = pd.read_csv(norm_path)
            kf_df = pd.read_csv(kf_path)
            
            if stage in kf_df.columns:
                target_frame = kf_df[stage].iloc[0]
                row = norm_df[norm_df['frame_id'] == target_frame].copy()
                
                if not row.empty:
                    # 필요 없는 컬럼 제거
                    row = row.drop(columns=['frame_id', 'timestamp'])
                    # 식별을 위한 정보 추가
                    row['stage'] = stage
                    row['source'] = f'GT{i}'
                    stage_data.append(row)

        if stage_data:
            # GT1~4 데이터 합치기
            stage_df = pd.concat(stage_data)
            
            # 2. 해당 단계의 평균(Average) 계산
            # numeric_only=True를 사용하여 문자열(stage, source) 제외하고 계산
            avg_values = stage_df.mean(numeric_only=True).to_frame().T
            avg_values['stage'] = stage
            avg_values['source'] = 'Average'
            
            # 3. GT 데이터들 다음에 평균 데이터 추가
            all_rows.append(stage_df)
            all_rows.append(avg_values)

    # 4. 전체 결과 통합 및 저장
    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        
        # 보기 좋게 컬럼 순서 조정 (stage, source를 맨 앞으로)
        cols = ['stage', 'source'] + [c for c in final_df.columns if c not in ['stage', 'source']]
        final_df = final_df[cols]
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 상세 데이터 포함 평균 추출 완료! 파일 저장 경로: {output_path}")
        return final_df
    else:
        print("❌ 데이터를 추출하지 못했습니다.")
        return None

# ==========================================
# 1. 경로 설정 (사용자 기존 경로 유지)
# ==========================================
normalized_csv_list = [
    '/Users/minji/Documents/minton-angle_resources/GT1_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT2_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT3_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT4_normalized_fixed.csv'
]

# ※ 참고: 현재 코드상 GT1.csv가 반복되고 있습니다. 실제 파일이 있다면 경로를 각각 수정해 주세요.
keyframe_csv_list = [
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT2.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT3.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT4.csv'
]

SAVE_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyframes_detailed.csv'

# ==========================================
# 2. 실행
# ==========================================
result = calculate_badminton_averages_with_details(normalized_csv_list, keyframe_csv_list, SAVE_PATH)

if result is not None:
    print(result.head(10)) # 결과 일부 확인