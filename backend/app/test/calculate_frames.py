import pandas as pd
import numpy as np

def calculate_badminton_averages(norm_files, kf_files, output_path):
    """
    norm_files: 4개의 정규화된 CSV 파일 경로 리스트 [GT1_norm, GT2_norm, ...]
    kf_files: 4개의 키프레임 CSV 파일 경로 리스트 [GT1_kf, GT2_kf, ...]
    output_path: 결과 평균 CSV 저장 경로
    """
    
    stages = ['ready', 'backswing', 'impact']
    # 각 단계별로 4개 영상의 데이터를 모을 딕셔너리
    collected_data = {stage: [] for stage in stages}

    # 4개의 영상 데이터를 순차적으로 처리
    for norm_path, kf_path in zip(norm_files, kf_files):
        # 데이터 불러오기
        norm_df = pd.read_csv(norm_path)
        kf_df = pd.read_csv(kf_path)
        
        for stage in stages:
            if stage in kf_df.columns:
                # 1. 해당 영상의 키프레임 번호 추출 (예: GT1의 ready 프레임은 27번)
                target_frame = kf_df[stage].iloc[0]
                
                # 2. 정규화된 데이터에서 해당 프레임의 행만 추출
                # frame_id와 timestamp는 평균 계산에서 제외
                row = norm_df[norm_df['frame_id'] == target_frame].drop(columns=['frame_id', 'timestamp'])
                
                if not row.empty:
                    collected_data[stage].append(row.iloc[0])

    # 3. 각 단계별 평균 계산
    average_rows = []
    for stage in stages:
        if collected_data[stage]:
            # 4개 영상의 데이터를 합쳐 DataFrame 생성 후 평균 계산
            stage_df = pd.DataFrame(collected_data[stage])
            stage_avg = stage_df.mean()
            stage_avg.name = stage  # 인덱스 이름을 단계명으로 설정
            average_rows.append(stage_avg)

    # 4. 결과 통합 및 저장
    if average_rows:
        final_df = pd.DataFrame(average_rows)
        final_df.to_csv(output_path, index_label='stage')
        print(f"✅ 평균 좌표 추출 완료! 파일 저장 경로: {output_path}")
        return final_df
    else:
        print("❌ 데이터를 추출하지 못했습니다. 경로를 확인해주세요.")
        return None

# ==========================================
# 1. 입력 경로 설정 (이 부분을 수정하세요)
# ==========================================

# 정규화된 좌표 CSV 파일 4개 경로
normalized_csv_list = [
    '/Users/minji/Documents/minton-angle_resources/GT1_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT2_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT3_normalized.csv', 
    '/Users/minji/Documents/minton-angle_resources/GT4_normalized_fixed.csv'
]

# 키프레임 번호가 적힌 CSV 파일 4개 경로
keyframe_csv_list = [
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv', 
    '/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1.csv'
]

# 결과 파일 저장 경로
SAVE_PATH = '/Users/minji/Documents/minton-angle/backend/data/standard/calculated_keyframes.csv'

# ==========================================
# 2. 실행
# ==========================================
result = calculate_badminton_averages(normalized_csv_list, keyframe_csv_list, SAVE_PATH)

# 결과 확인
if result is not None:
    print(result)