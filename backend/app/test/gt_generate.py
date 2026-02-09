import cv2
import mediapipe as mp
import pandas as pd
import sys
import time
import os

print("=" * 60)
print("GT 영상 → Standard Keypoint CSV 생성")
print("=" * 60)

# ========================================
# 선택된 Keypoint 정의 (19개)
# ========================================
SELECTED_KEYPOINTS = {
    0: 'nose',
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow', 14: 'right_elbow',
    15: 'left_wrist', 16: 'right_wrist',
    17: 'left_pinky', 18: 'right_pinky',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle',
    29: 'left_heel', 30: 'right_heel',
    31: 'left_foot_index', 32: 'right_foot_index'
}

print(f"\n선택된 Keypoint: {len(SELECTED_KEYPOINTS)}개")
print("  - 얼굴: 1개 (nose)")
print("  - 손: 4개 (wrist, pinky)")
print("  - 상/하체: 8개 (shoulder, elbow, hip, knee)")
print("  - 발: 6개 (ankle, heel, foot_index)")

# ========================================
# 1. MediaPipe 설정
# ========================================
print("\n[1/7] AI 모델 로딩 중...")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe 모델 로드 완료!")

# ========================================
# 2. 출력 폴더 설정 (수정!)
# ========================================
print("\n[2/7] 출력 폴더 설정...")

# 현재 파일 위치: backend/app/test/gt_generate.py
current_dir = os.path.dirname(os.path.abspath(__file__))  # backend/app/test
app_dir = os.path.dirname(current_dir)                     # backend/app
backend_dir = os.path.dirname(app_dir)                     # backend ✅
output_dir = os.path.join(backend_dir, 'data', 'standard') # backend/data/standard ✅

print(f"   현재 위치: {current_dir}")
print(f"   Backend: {backend_dir}")
print(f"   저장 경로: {output_dir}")

# 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ 폴더 생성: {output_dir}")
else:
    print(f"✅ 폴더 확인: {output_dir}")

# ========================================
# 3. 동영상 파일 입력
# ========================================
print("\n[3/7] 전문가 GT 영상 불러오기...")
print("\n💡 전문가 영상 경로를 입력하세요:")
print("   (유튜브 전문가 영상 또는 고품질 영상)")
print()

video_path = input("GT 영상 경로: ").strip().strip('"').strip("'")

if not os.path.exists(video_path):
    print(f"\n❌ 파일을 찾을 수 없습니다: {video_path}")
    sys.exit()

# ========================================
# 4. 동영상 열기
# ========================================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"\n❌ 동영상을 열 수 없습니다!")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps

print(f"\n✅ 동영상 로드 완료!")
print(f"   파일: {os.path.basename(video_path)}")
print(f"   해상도: {width}×{height}")
print(f"   FPS: {fps:.2f}")
print(f"   총 프레임: {total_frames}개")
print(f"   길이: {duration:.2f}초")

# ========================================
# 5. Keypoint 추출
# ========================================
print(f"\n[4/7] Keypoint 추출 중...")

keypoints_list = []
frame_id = 0
detected_count = 0
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        frame_data = {
            'frame_id': frame_id,
            'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        }
        
        for idx, name in SELECTED_KEYPOINTS.items():
            landmark = results.pose_landmarks.landmark[idx]
            frame_data[f'{name}_x'] = landmark.x
            frame_data[f'{name}_y'] = landmark.y
            frame_data[f'{name}_z'] = landmark.z
            frame_data[f'{name}_visibility'] = landmark.visibility
        
        keypoints_list.append(frame_data)
        detected_count += 1
    
    if frame_id % 30 == 0:
        progress = (frame_id / total_frames) * 100
        elapsed = time.time() - start_time
        remaining = (elapsed / (frame_id + 1)) * (total_frames - frame_id)
        print(f"   진행: {frame_id}/{total_frames} ({progress:.1f}%) | 남은 시간: {int(remaining)}초")
    
    frame_id += 1

cap.release()

elapsed_time = time.time() - start_time
print(f"\n✅ Keypoint 추출 완료!")
print(f"   처리 시간: {elapsed_time:.2f}초")
print(f"   총 프레임: {frame_id}개")
print(f"   검출 성공: {detected_count}개 ({detected_count/frame_id*100:.1f}%)")

if detected_count == 0:
    print("\n❌ 사람을 검출하지 못했습니다!")
    sys.exit()

# ========================================
# 6. CSV 저장 (data/standard/)
# ========================================
print(f"\n[5/7] CSV 저장 중...")

df = pd.DataFrame(keypoints_list)

# 파일명 입력 받기
print("\n💡 저장할 파일명을 입력하세요:")
print("   예: clear_pro, smash_expert, drop_professional")
file_name = input("파일명 (확장자 제외): ").strip()

if not file_name:
    file_name = os.path.splitext(os.path.basename(video_path))[0]

output_csv = os.path.join(output_dir, f"{file_name}_gt.csv")

df.to_csv(output_csv, index=False)

print(f"\n✅ CSV 저장 완료!")
print(f"   경로: {output_csv}")
print(f"   Shape: {df.shape}")
print(f"   컬럼: {len(df.columns)}개")

# 7. 프레임 이미지 저장 (선택)
# ========================================
# gt_generate.py에 추가

# ========================================
# 프레임 이미지 저장 함수
# ========================================

def save_frame_images(video_path, output_folder, file_name, save_interval=1):
    """
    동영상에서 프레임별 이미지 저장
    
    Args:
        video_path: 동영상 경로
        output_folder: 이미지 저장 폴더
        file_name: 파일명 prefix
        save_interval: 저장 간격 (1=모든 프레임, 5=5프레임마다)
    """
    
    print("\n" + "=" * 60)
    print("프레임 이미지 저장")
    print("=" * 60)
    
    # 이미지 저장 폴더 생성
    image_folder = os.path.join(output_folder, f"{file_name}_frames")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    print(f"\n📂 이미지 저장 폴더: {image_folder}")
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_id = 0
    saved_count = 0
    
    print(f"\n💾 이미지 저장 중... (간격: {save_interval}프레임)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 저장 간격에 맞으면 저장
        if frame_id % save_interval == 0:
            image_path = os.path.join(image_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"   저장: {saved_count}개 ({progress:.1f}%)")
        
        frame_id += 1
    
    cap.release()
    
    print(f"\n✅ 이미지 저장 완료!")
    print(f"   총 {saved_count}개 이미지")
    print(f"   폴더: {image_folder}")
    
    return image_folder


# ========================================
# Keypoint 표시된 이미지 저장 함수
# ========================================

def save_keypoint_images(video_path, keypoints_list, output_folder, file_name, 
                         save_interval=1, draw_skeleton=True):
    """
    Keypoint를 표시한 이미지 저장
    
    Args:
        video_path: 동영상 경로
        keypoints_list: 추출한 keypoint 리스트
        output_folder: 저장 폴더
        file_name: 파일명 prefix
        save_interval: 저장 간격
        draw_skeleton: 스켈레톤 그리기 여부
    """
    
    print("\n" + "=" * 60)
    print("Keypoint 시각화 이미지 저장")
    print("=" * 60)
    
    # 폴더 생성
    image_folder = os.path.join(output_folder, f"{file_name}_keypoints")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    print(f"\n📂 이미지 저장 폴더: {image_folder}")
    
    # MediaPipe 설정
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_id = 0
    saved_count = 0
    
    print(f"\n💾 Keypoint 이미지 저장 중... (간격: {save_interval}프레임)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_id % save_interval == 0:
            # MediaPipe 처리
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Keypoint 그리기
            if results.pose_landmarks:
                if draw_skeleton:
                    # 스켈레톤 그리기
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                else:
                    # 점만 그리기
                    h, w = frame.shape[:2]
                    for landmark in results.pose_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # 프레임 번호 표시
            cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 저장
            image_path = os.path.join(image_folder, f"frame_{frame_id:04d}_kp.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"   저장: {saved_count}개 ({progress:.1f}%)")
        
        frame_id += 1
    
    cap.release()
    pose.close()
    
    print(f"\n✅ Keypoint 이미지 저장 완료!")
    print(f"   총 {saved_count}개 이미지")
    print(f"   폴더: {image_folder}")
    
    return image_folder

print(f"\n[6/8] 프레임 이미지 저장 (선택사항)")

print("\n❓ 프레임 이미지를 저장할까요?")
print("   1. 원본 프레임만 저장")
print("   2. Keypoint 표시된 프레임 저장")
print("   3. 둘 다 저장")
print("   4. 저장 안 함")

choice = input("\n선택 (1/2/3/4): ").strip()

if choice in ['1', '2', '3']:
    print("\n💡 저장 간격 설정:")
    print("   1 = 모든 프레임")
    print("   5 = 5프레임마다")
    print("   10 = 10프레임마다")
    
    interval = input("\n저장 간격 (기본 1): ").strip()
    save_interval = int(interval) if interval.isdigit() else 1
    
    if choice in ['1', '3']:
        # 원본 프레임 저장
        save_frame_images(video_path, output_dir, file_name, save_interval)
    
    if choice in ['2', '3']:
        # Keypoint 프레임 저장
        save_keypoint_images(video_path, keypoints_list, output_dir, 
                           file_name, save_interval, draw_skeleton=True)
else:
    print("\n⏭️  이미지 저장 건너뛰기")

# ========================================
# 8. 데이터 미리보기
# ========================================
print(f"\n[6/7] 데이터 미리보기:")
print(df[['frame_id', 'timestamp', 'nose_x', 'left_elbow_y', 'right_wrist_y']].head())

# ========================================
# 9. 통계 정보
# ========================================
print(f"\n[7/7] 통계 정보:")

visibility_cols = [f'{name}_visibility' for name in SELECTED_KEYPOINTS.values()]
avg_visibility = df[visibility_cols].mean().mean()

print(f"   평균 가시성: {avg_visibility:.3f}")
print(f"   검출률: {detected_count}/{frame_id} ({detected_count/frame_id*100:.1f}%)")

groups = {
    '얼굴': ['nose'],
    '손': ['left_wrist', 'right_wrist', 'left_pinky', 'right_pinky'],
    '상체': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],
    '하체': ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
    '발': ['left_ankle', 'right_ankle', 'left_heel', 'right_heel', 
           'left_foot_index', 'right_foot_index']
}

print(f"\n   그룹별 가시성:")
for group_name, keypoints in groups.items():
    vis_cols = [f'{kp}_visibility' for kp in keypoints]
    group_vis = df[vis_cols].mean().mean()
    status = "✅" if group_vis > 0.7 else "⚠️" if group_vis > 0.5 else "❌"
    print(f"   {status} {group_name:6s}: {group_vis:.3f}")

print("\n" + "=" * 60)
print("GT Standard 데이터 생성 완료!")
print("=" * 60)
print(f"\n📁 저장 위치:")
print(f"   {output_csv}")
print(f"\n📂 폴더 구조:")
print(f"   backend/")
print(f"   └── data/")
print(f"       └── standard/")
print(f"           └── {file_name}_gt.csv  ← 전문가 GT")
print(f"\n💡 이 GT는 모든 사용자 분석에 사용됩니다!")
print(f"\n🎯 다음 단계:")
print("  1. 사용자 영상 처리 (data/video/)")
print("  2. GT vs User 비교 (FastDTW)")
print("  3. 자세 진단 & 피드백")