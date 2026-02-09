import cv2
import numpy as np
import pandas as pd
import math
import os

class BadmintonAnalyzer:
    def __init__(self):
        # 점수 기준 및 가중치 설정
        self.weights = {'elbow': 0.4, 'backswing': 0.3, 'rotation': 0.3}
        
    def calculate_angle(self, p1, p2, p3):
        """세 점 사이의 각도를 계산 (p2가 정점)"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def get_rotation_angle(self, left_shoulder, right_shoulder):
        """양 어깨 선의 기울기 각도 계산"""
        dy = right_shoulder[1] - left_shoulder[1]
        dx = right_shoulder[0] - left_shoulder[0]
        angle = math.degrees(math.atan2(dy, dx))
        return abs(angle)

    def evaluate_score(self, keypoints):
        """3가지 기준에 따른 점수 및 피드백 생성"""
        # 편의를 위해 인덱스 정의 (MediaPipe 기준 예시)
        # 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow, 15: L_Wrist, 16: R_Wrist
        # 사용자의 데이터 구조에 맞게 수정 필요
        l_sh, r_sh = keypoints[11], keypoints[12]
        l_el, r_el = keypoints[13], keypoints[14]
        l_wr, r_wr = keypoints[15], keypoints[16]

        # 오른손잡이 기준 (필요시 왼손 로직 추가)
        # 1. 팔꿈치 높이 (어깨 Y - 팔꿈치 Y) -> OpenCV는 아래로 갈수록 Y 증가하므로 반대로 뺌
        elbow_height = r_sh[1] - r_el[1] 
        if elbow_height >= 0.15: s1, status1 = 100, "✅ 좋아요"
        elif 0.05 <= elbow_height < 0.15: s1, status1 = 70, "⚠️ 아쉬워요"
        else: s1, status1 = 40, "❌ 나빠요"

        # 2. 백스윙 각도
        backswing_angle = self.calculate_angle(r_sh, r_el, r_wr)
        if 60 <= backswing_angle <= 90: s2, status2 = 100, "✅ 좋아요"
        elif 40 <= backswing_angle < 60: s2, status2 = 70, "⚠️ 아쉬워요"
        else: s2, status2 = 40, "❌ 나빠요"

        # 3. 몸통 회전
        rotation_angle = self.get_rotation_angle(l_sh, r_sh)
        if 45 <= rotation_angle <= 60: s3, status3 = 100, "✅ 좋아요"
        elif 30 <= rotation_angle < 45: s3, status3 = 70, "⚠️ 아쉬워요"
        else: s3, status3 = 40, "❌ 나빠요"

        total_score = (s1 * self.weights['elbow']) + (s2 * self.weights['backswing']) + (s3 * self.weights['rotation'])
        
        feedback = {
            'total': round(total_score, 1),
            'details': [
                {'part': '팔꿈치 높이', 'score': s1, 'status': status1, 'val': f"{elbow_height:.2f}", 'desc': "어깨 대비 팔꿈치 위치"},
                {'part': '백스윙 각도', 'score': s2, 'status': status2, 'val': f"{backswing_angle:.1f}°", 'desc': "팔의 굽힘 정도"},
                {'part': '몸통 회전', 'score': s3, 'status': status3, 'val': f"{rotation_angle:.1f}°", 'desc': "어깨선의 회전각"}
            ]
        }
        return feedback

    def draw_skeleton(self, image, keypoints, color=(0, 255, 0)):
        """이미지에 주요 키포인트와 연결선 그리기"""
        # 주요 연결 부위 (어깨-어깨, 어깨-팔꿈치, 팔꿈치-손목 등)
        connections = [(11, 12), (12, 14), (14, 16), (11, 13), (13, 15)]
        h, w, _ = image.shape
        
        for start_idx, end_idx in connections:
            pt1 = (int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h))
            pt2 = (int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h))
            cv2.line(image, pt1, pt2, color, 3)
            
        for kp in keypoints:
            cv2.circle(image, (int(kp[0] * w), int(kp[1] * h)), 5, (0, 0, 255), -1)
        return image

    def process_comparison(self, gt_img_path, user_img_path, gt_kps, user_kps, frame_name):
        # 1. 이미지 로드 및 스켈레톤 그리기
        img_gt = cv2.imread(gt_img_path)
        img_user = cv2.imread(user_img_path)
        
        # 이미지 크기 통일 (비교를 위해 가로 길이를 맞춤)
        img_user = cv2.resize(img_user, (img_gt.shape[1], img_gt.shape[0]))
        
        img_gt = self.draw_skeleton(img_gt, gt_kps, color=(255, 0, 0)) # GT는 파란색
        img_user = self.draw_skeleton(img_user, user_kps, color=(0, 255, 0)) # 유저는 초록색

        # 2. 점수 계산
        result = self.evaluate_score(user_kps)

        # 3. 결과 텍스트 합성
        combined_img = cv2.hconcat([img_gt, img_user])
        cv2.putText(combined_img, f"Score: {result['total']}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        return combined_img, result

# --- 실행부 ---
analyzer = BadmintonAnalyzer()

# 예시 데이터 (실제 프로젝트에서는 추출된 좌표 리스트를 넣으세요)
# 키포인트 예시: [[x1, y1], [x2, y2], ... 33개] (정규화된 0~1 사이 값)
mock_user_kps = np.random.rand(33, 2) * 0.5 + 0.2 # 가상 데이터
mock_gt_kps = np.random.rand(33, 2) * 0.5 + 0.2

# 파일 경로 (실제 경로로 수정 필요)
gt_path = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/GT1/GT1_normalized_ready.jpg"
user_path = "/Users/minji/Documents/minton-angle/backend/data/standard/final_gt_frames/roh/roh_normalized_fixed_ready.jpg"

if os.path.exists(gt_path) and os.path.exists(user_path):
    final_img, score_data = analyzer.process_comparison(gt_path, user_path, mock_gt_kps, mock_user_kps, "Frame_01")
    
    # 1. 이미지 저장
    cv2.imwrite("/Users/minji/Documents/minton-angle/backend/data/standard/calculated_score/comparison_result.jpg", final_img)
    print("비교 이미지 저장 완료: comparison_result.jpg")

    # 2. CSV 리포트 생성
    report_list = []
    for detail in score_data['details']:
        report_list.append({
            '키프레임': "Frame_01",
            '총점': score_data['total'],
            '평가항목': detail['part'],
            '상태': detail['status'],
            '측정값': detail['val'],
            '피드백': detail['desc']
        })
    
    df = pd.DataFrame(report_list)
    df.to_csv("/Users/minji/Documents/minton-angle/backend/data/standard/calculated_score/comparison_report.csv", index=False, encoding='utf-8-sig')
    print("CSV 리포트 저장 완료: comparison_report.csv")