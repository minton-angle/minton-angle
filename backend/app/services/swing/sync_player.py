"""
전문가 영상 키프레임 찾기 (한 명씩 순서대로)

실행: python find_keyframes.py

조작:
  - Space: 재생/일시정지
  - A/D 또는 ←/→: 1프레임 이동
  - W/S 또는 [/]: 10프레임 이동
  - 1: E1 (준비자세) 저장
  - 2: E2 (백스윙) 저장  
  - 3: E3 (임팩트) 저장
  - Enter: 다음 영상으로
  - Q: 종료 & 저장
"""

import cv2
import csv
from pathlib import Path

VIDEO_DIR = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\expert_videos")
OUTPUT_PATH = Path(r"C:\GitHub_Project\AICV_03\minton-angle\backend\data\standard\keyframe_labels.csv")


def find_keyframes():
    """한 명씩 키프레임 찾기"""
    
    # 영상 파일 찾기
    video_files = sorted(VIDEO_DIR.glob("expert_*.mp4"))
    if not video_files:
        video_files = sorted(VIDEO_DIR.glob("expert_*.MP4"))
    
    print(f"📁 영상 {len(video_files)}개 발견")
    print("\n" + "=" * 50)
    print("🎮 조작법:")
    print("   Space: 재생/일시정지")
    print("   A / D: 1프레임 이동 (← →)")
    print("   W / S: 10프레임 이동")
    print("   1: E1 (준비자세) 저장")
    print("   2: E2 (백스윙) 저장")
    print("   3: E3 (임팩트) 저장")
    print("   Enter: 다음 영상으로")
    print("   Q: 전체 종료 & 저장")
    print("=" * 50)
    
    # 결과 저장
    all_results = []
    
    for idx, video_path in enumerate(video_files):
        expert_id = video_path.stem
        
        print(f"\n{'─' * 50}")
        print(f"[{idx + 1}/{len(video_files)}] {expert_id}")
        print(f"{'─' * 50}")
        
        result = process_single_video(video_path, expert_id, idx + 1, len(video_files))
        
        if result is None:  # Q 눌러서 종료
            break
        
        all_results.append(result)
        print(f"✅ {expert_id} 완료: E1={result['E1']}, E2={result['E2']}, E3={result['E3']}")
    
    # CSV 저장
    save_results(all_results)
    
    cv2.destroyAllWindows()


def process_single_video(video_path, expert_id, current_num, total_num):
    """단일 영상 처리"""
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 화면 크기 조정
    scale = min(1.0, 1200 / width, 800 / height)
    disp_w = int(width * scale)
    disp_h = int(height * scale)
    
    current_frame = 0
    playing = False
    keyframes = {'E1': None, 'E2': None, 'E3': None}
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            current_frame = max(0, current_frame - 1)
            continue
        
        # 리사이즈
        display = cv2.resize(frame, (disp_w, disp_h))
        
        # 상단 정보 패널
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, 120), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        
        # 텍스트 정보
        cv2.putText(display, f"{expert_id} ({current_num}/{total_num})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {current_frame} / {total_frames - 1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 키프레임 상태
        e1_text = f"E1(준비): {keyframes['E1']}" if keyframes['E1'] is not None else "E1(준비): -"
        e2_text = f"E2(백스윙): {keyframes['E2']}" if keyframes['E2'] is not None else "E2(백스윙): -"
        e3_text = f"E3(임팩트): {keyframes['E3']}" if keyframes['E3'] is not None else "E3(임팩트): -"
        
        cv2.putText(display, e1_text, (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if keyframes['E1'] else (100, 100, 100), 2)
        cv2.putText(display, e2_text, (200, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if keyframes['E2'] else (100, 100, 100), 2)
        cv2.putText(display, e3_text, (420, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if keyframes['E3'] else (100, 100, 100), 2)
        
        # 하단 안내
        guide = "[Space:Play] [A/D:1F] [W/S:10F] [1:E1] [2:E2] [3:E3] [Enter:Next]"
        cv2.putText(display, guide, (10, disp_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # 재생 상태 표시
        if playing:
            cv2.putText(display, "PLAYING", (disp_w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, "PAUSED", (disp_w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Find Keyframes", display)
        
        # ✅ waitKeyEx 사용 (방향키 인식)
        key = cv2.waitKeyEx(30 if playing else 0)
        
        # Q: 종료
        if key == ord('q') or key == ord('Q'):
            cap.release()
            return None
        
        # Enter: 다음 영상
        elif key == 13:
            if keyframes['E1'] is not None and keyframes['E2'] is not None and keyframes['E3'] is not None:
                cap.release()
                return {
                    'expert_id': expert_id,
                    'E1': keyframes['E1'],
                    'E2': keyframes['E2'],
                    'E3': keyframes['E3']
                }
            else:
                print("⚠️ E1, E2, E3 모두 저장해야 다음으로 넘어갈 수 있어요!")
        
        # Space: 재생/정지
        elif key == ord(' '):
            playing = not playing
        
        # ✅ 1프레임 뒤로: A 또는 왼쪽 화살표
        elif key == ord('a') or key == ord('A') or key == 2424832:
            current_frame = max(0, current_frame - 1)
            playing = False
        
        # ✅ 1프레임 앞으로: D 또는 오른쪽 화살표
        elif key == ord('d') or key == ord('D') or key == 2555904:
            current_frame = min(total_frames - 1, current_frame + 1)
            playing = False
        
        # ✅ 10프레임 뒤로: W 또는 [
        elif key == ord('w') or key == ord('W') or key == ord('['):
            current_frame = max(0, current_frame - 10)
            playing = False
        
        # ✅ 10프레임 앞으로: S 또는 ]
        elif key == ord('s') or key == ord('S') or key == ord(']'):
            current_frame = min(total_frames - 1, current_frame + 10)
            playing = False
        
        # 1: E1 저장
        elif key == ord('1'):
            keyframes['E1'] = current_frame
            print(f"   ✅ E1 (준비자세): Frame {current_frame}")
        
        # 2: E2 저장
        elif key == ord('2'):
            keyframes['E2'] = current_frame
            print(f"   ✅ E2 (백스윙): Frame {current_frame}")
        
        # 3: E3 저장
        elif key == ord('3'):
            keyframes['E3'] = current_frame
            print(f"   ✅ E3 (임팩트): Frame {current_frame}")
        
        # 자동 재생
        if playing:
            current_frame = min(total_frames - 1, current_frame + 1)
            if current_frame >= total_frames - 1:
                playing = False


def save_results(results):
    """결과 CSV 저장"""
    
    if not results:
        print("\n❌ 저장된 결과 없음")
        return
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['expert_id', 'E1_ready', 'E2_backswing', 'E3_impact'])
        
        for r in results:
            writer.writerow([r['expert_id'], r['E1'], r['E2'], r['E3']])
    
    print("\n" + "=" * 50)
    print("📋 저장 완료!")
    print("=" * 50)
    
    for r in results:
        print(f"   {r['expert_id']}: E1={r['E1']}, E2={r['E2']}, E3={r['E3']}")
    
    print(f"\n✅ 파일: {OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    find_keyframes()