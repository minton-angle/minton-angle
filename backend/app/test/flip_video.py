import cv2
import os

# --- 1. 경로 설정 ---
INPUT_VIDEO = '/Users/minji/Documents/pro_swing_3_GT.mp4'    # 원본 영상 파일명
OUTPUT_VIDEO = '/Users/minji/Documents/pro_swing_3_GT.mp4/pro_swing_3.GT_flip.mp4'     # 저장할 결과 파일명

# --- 2. 영상 불러오기 ---
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print(f"❌ 오류: 영상을 열 수 없습니다. 경로를 확인하세요: {INPUT_VIDEO}")
    exit()

# 원본 영상의 속성 가져오기 (결과 영상도 똑같이 맞추기 위함)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 저장을 위한 코덱

# --- 3. 영상 저장 객체 생성 ---
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"🔄 영상 반전 시작: {INPUT_VIDEO} -> {OUTPUT_VIDEO}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 좌우 반전 (1: 좌우, 0: 상하, -1: 상하좌우)
    flipped_frame = cv2.flip(frame, 1)

    # 반전된 프레임을 결과 파일에 쓰기
    out.write(flipped_frame)

    # (선택) 처리 과정을 화면으로 보고 싶다면 아래 주석 해제
    # cv2.imshow('Flipping...', flipped_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"🎉 성공! 반전된 영상이 저장되었습니다: {os.path.abspath(OUTPUT_VIDEO)}")