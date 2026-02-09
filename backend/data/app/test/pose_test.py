import cv2
import mediapipe as mp
import sys
import time

print(">>> [1/5] 라이브러리 로딩 중...")

# 1. MediaPipe Pose 모델 설정
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,      # 가장 가벼운 0단계로 수정했습니다!
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    print(">>> [2/5] AI 모델 생성 성공!")
except Exception as e:
    print(f">>> [에러] AI 모델 생성 실패: {e}")
    sys.exit()

# 2. 웹캠 연결 시도
print(">>> [3/5] 카메라 연결 시도 중...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(">>> [에러] 0번 카메라를 열 수 없습니다.")
    sys.exit()

print(">>> [4/5] 카메라 연결 성공! 프레임을 읽기 시작합니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(">>> [알림] 프레임을 읽지 못해 대기 중...")
        continue

    # RGB 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 분석 수행
    start_time = time.time()
    results = pose.process(image)
    end_time = time.time()

    # 3. 화면 표시
    if results.pose_landmarks:
        # 화면에 뼈대 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 분석 속도를 화면에 표시 (정상 작동 확인용)
        fps = 1.0 / (end_time - start_time)
        cv2.putText(frame, f"AI FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MintonAngle Debug Mode', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

print(">>> [5/5] 프로그램 종료 중...")
cap.release()
cv2.destroyAllWindows()