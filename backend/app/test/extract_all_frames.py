"""
GT 동영상의 모든 프레임을 이미지로 추출
- 3개 GT 동영상 → 각각 모든 프레임 저장
- 프레임 번호 확인용
"""

import cv2
import os
import sys


def extract_all_frames(video_path: str, output_folder: str, 
                      frame_interval: int = 1, add_frame_number: bool = True):
    """
    동영상의 모든 프레임을 이미지로 저장
    
    Args:
        video_path: 동영상 파일 경로
        output_folder: 이미지 저장 폴더
        frame_interval: 저장 간격 (1=모든 프레임, 5=5프레임마다)
        add_frame_number: 이미지에 프레임 번호 표시 여부
    """
    
    print("=" * 60)
    print(f"📹 동영상 프레임 추출")
    print("=" * 60)
    print(f"\n입력: {video_path}")
    print(f"출력: {output_folder}")
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"\n❌ 동영상을 열 수 없습니다!")
        return
    
    # 동영상 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📊 동영상 정보:")
    print(f"   총 프레임: {total_frames}개")
    print(f"   FPS: {fps:.2f}")
    print(f"   해상도: {width}x{height}")
    print(f"   길이: {total_frames/fps:.2f}초")
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n💾 프레임 추출 중...")
    print(f"   저장 간격: {frame_interval}프레임마다")
    
    saved_count = 0
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 지정된 간격마다 저장
        if frame_id % frame_interval == 0:
            
            # 프레임 번호 표시 (선택)
            if add_frame_number:
                display_frame = frame.copy()
                
                # 상단에 검은 배경
                cv2.rectangle(display_frame, (0, 0), (300, 60), (0, 0, 0), -1)
                
                # 프레임 번호 텍스트
                timestamp = frame_id / fps
                text = f"Frame: {frame_id}"
                time_text = f"Time: {timestamp:.2f}s"
                
                cv2.putText(display_frame, text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, time_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                display_frame = frame
            
            # 이미지 저장
            filename = f"frame_{frame_id:04d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, display_frame)
            
            saved_count += 1
            
            # 진행 상황 출력 (10프레임마다)
            if saved_count % 10 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"   진행: {progress:.1f}% ({frame_id}/{total_frames} 프레임)")
        
        frame_id += 1
    
    cap.release()
    
    print(f"\n✅ 추출 완료!")
    print(f"   총 저장: {saved_count}개 이미지")
    print(f"   폴더: {output_folder}")
    print(f"\n💡 이제 이미지를 보고 KF1, KF2, KF3 프레임 번호를 확인하세요!")


def extract_multiple_videos(video_paths: list, base_output_folder: str,
                           frame_interval: int = 1):
    """
    여러 동영상의 프레임을 각각 추출
    
    Args:
        video_paths: 동영상 경로 리스트
        base_output_folder: 기본 출력 폴더
        frame_interval: 저장 간격
    """
    
    print("=" * 60)
    print(f"📹 여러 GT 동영상 프레임 추출")
    print("=" * 60)
    print(f"\n총 {len(video_paths)}개 동영상\n")
    
    for i, video_path in enumerate(video_paths, 1):
        print("\n" + "=" * 60)
        print(f"[{i}/{len(video_paths)}] {os.path.basename(video_path)}")
        print("=" * 60)
        
        # 동영상별 출력 폴더
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(base_output_folder, f"{video_name}_frames")
        
        extract_all_frames(video_path, output_folder, frame_interval)
    
    print("\n" + "=" * 60)
    print("🎉 모든 동영상 추출 완료!")
    print("=" * 60)
    print(f"\n📁 저장 위치: {base_output_folder}")
    print(f"\n💡 다음 단계:")
    print(f"   1. 각 폴더의 이미지를 보고 Key Frame 확인")
    print(f"   2. KF1, KF2, KF3 프레임 번호 메모")
    print(f"   3. 평균 패턴 학습 스크립트 실행")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("GT 동영상 프레임 추출 도구")
    print("=" * 60)
    
    # 모드 선택
    print("\n💡 실행 모드:")
    print("   1. 단일 동영상")
    print("   2. 여러 동영상 (GT 3개)")
    
    mode = input("\n선택 (1/2): ").strip()
    
    if mode == '2':
        # ===== 여러 동영상 =====
        print("\n" + "=" * 60)
        print("📹 3개 GT 동영상 입력")
        print("=" * 60)
        
        video_paths = []
        
        for i in range(1, 4):
            print(f"\n[GT {i}/3]")
            video_path = input(f"동영상 경로: ").strip().strip('"').strip("'")
            
            if not os.path.exists(video_path):
                print(f"⚠️  파일을 찾을 수 없습니다: {video_path}")
                continue
            
            video_paths.append(video_path)
            print(f"✅ {os.path.basename(video_path)}")
        
        if len(video_paths) == 0:
            print("\n❌ 유효한 동영상이 없습니다!")
            sys.exit()
        
        print(f"\n✅ 총 {len(video_paths)}개 동영상 입력됨")
        
        # 출력 폴더
        print("\n💡 출력 폴더 경로:")
        print("   (예: C:\\badminton\\gt_frames)")
        
        output_folder = input("\n출력 폴더: ").strip().strip('"').strip("'")
        
        if not output_folder:
            # 첫 번째 동영상과 같은 폴더에 생성
            output_folder = os.path.join(os.path.dirname(video_paths[0]), "gt_all_frames")
            print(f"\n✅ 기본 폴더 사용: {output_folder}")
        
        # 저장 간격
        print("\n💡 프레임 저장 간격:")
        print("   1 = 모든 프레임 (권장)")
        print("   5 = 5프레임마다")
        print("   10 = 10프레임마다")
        
        interval_input = input("\n간격 (Enter=1): ").strip()
        frame_interval = int(interval_input) if interval_input else 1
        
        print(f"\n✅ {frame_interval}프레임마다 저장")
        
        # 추출 시작
        extract_multiple_videos(video_paths, output_folder, frame_interval)
    
    else:
        # ===== 단일 동영상 =====
        print("\n💡 동영상 파일 경로:")
        video_path = input("동영상 경로: ").strip().strip('"').strip("'")
        
        if not os.path.exists(video_path):
            print(f"\n❌ 파일을 찾을 수 없습니다!")
            sys.exit()
        
        # 출력 폴더
        print("\n💡 출력 폴더 경로:")
        print("   (Enter = 동영상과 같은 폴더)")
        
        output_folder = input("출력 폴더: ").strip().strip('"').strip("'")
        
        if not output_folder:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_dir = os.path.dirname(video_path)
            output_folder = os.path.join(video_dir, f"{video_name}_frames")
            print(f"\n✅ 기본 폴더: {output_folder}")
        
        # 저장 간격
        print("\n💡 프레임 저장 간격:")
        print("   1 = 모든 프레임")
        print("   5 = 5프레임마다")
        
        interval_input = input("\n간격 (Enter=1): ").strip()
        frame_interval = int(interval_input) if interval_input else 1
        
        # 프레임 번호 표시
        print("\n💡 이미지에 프레임 번호 표시:")
        print("   1. 표시 (권장)")
        print("   2. 표시 안함")
        
        show_number = input("\n선택 (1/2): ").strip()
        add_frame_number = (show_number != '2')
        
        # 추출 시작
        extract_all_frames(video_path, output_folder, frame_interval, add_frame_number)