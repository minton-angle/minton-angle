import cv2
import mediapipe as mp
import numpy as np

class PoseEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0보다 1이 조금 더 정확합니다 (배드민턴은 정교해야 하니까요!)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """세 점(a, b, c)을 받아 b를 정점으로 하는 각도를 계산 (degree)"""
        a = np.array(a) # 첫 번째 점 (예: 어깨)
        b = np.array(b) # 두 번째 점 (예: 팔꿈치)
        c = np.array(c) # 세 번째 점 (예: 손목)

        # 벡터 계산
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        """프레임 하나를 받아 관절 좌표와 계산된 각도를 반환"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        analysis_result = {
            "landmarks": None,
            "elbow_angle": 0,
            "wrist_coord": None
        }

        if results.pose_landmarks:
            analysis_result["landmarks"] = results.pose_landmarks
            landmarks = results.pose_landmarks.landmark

            # 배드민턴 핵심 관절: 오른쪽 어깨(12), 팔꿈치(14), 손목(16)
            shoulder = [landmarks[12].x, landmarks[12].y]
            elbow = [landmarks[14].x, landmarks[14].y]
            wrist = [landmarks[16].x, landmarks[16].y]

            # 팔꿈치 각도 계산
            analysis_result["elbow_angle"] = self.calculate_angle(shoulder, elbow, wrist)
            analysis_result["wrist_coord"] = wrist

        return analysis_result, results

# 사용 예시 (테스트용)
if __name__ == "__main__":
    engine = PoseEngine()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        result, raw_results = engine.process_frame(frame)
        
        if result["landmarks"]:
            # 뼈대 그리기
            engine.mp_drawing.draw_landmarks(frame, result["landmarks"], engine.mp_pose.POSE_CONNECTIONS)
            # 각도 표시
            cv2.putText(frame, f"Angle: {int(result['elbow_angle'])}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('MintonAngle Core Test', frame)
        if cv2.waitKey(5) & 0xFF == 27: break
    cap.release()