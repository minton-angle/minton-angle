"""
배드민턴 스윙 6대 핵심 지표 계산 모듈
- 준비자세(KF1): 양발 거리, 라켓 손 높이, 무릎 각도
- 백스윙(KF2): 팔꿈치 높이, 어깨 회전각, 왼손 위치
- 임팩트(KF3): 타구 높이, 팔 직진성, 몸 기울기
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pandas as pd


class BadmintonMetrics:
    """배드민턴 스윙 6대 지표 계산 및 진단"""
    
    def __init__(self):
        # 기준값 설정 (논문 + 실용성 기반)
        self.thresholds = {
            # ===== 준비 자세 (KF1) =====
            'foot_distance': {
                'good': (0.15, 0.25),    # 어깨 너비 1~1.5배
                'ok': (0.10, 0.30),
                'bad': 'outside'
            },
            'racket_hand_height': {
                'good': (0.05, 0.15),     # 어깨보다 5~15cm 위
                'ok': (0, 0.20),
                'bad': 'outside'
            },
            'knee_angle': {
                'good': (150, 170),       # 약간 굽힘
                'ok': (140, 175),
                'bad': 'outside'
            },
            
            # ===== 백스윙 (KF2) =====
            'elbow_height': {
                'good': (0.15, 0.25),     # 어깨보다 15~25cm 위
                'ok': (0.05, 0.30),
                'bad': 'outside'
            },
            'shoulder_rotation': {
                'good': (45, 60),         # 45~60도 회전
                'ok': (30, 70),
                'bad': 'outside'
            },
            'left_hand_position': {
                'good': (0.10, 0.25),     # 셔틀콕 방향
                'ok': (0.05, 0.30),
                'bad': 'outside'
            },
            
            # ===== 임팩트 (KF3) =====
            'impact_height': {
                'good': (0.85, 0.95),     # 키의 85~95%
                'ok': (0.75, 1.00),
                'bad': 'outside'
            },
            'arm_straightness': {
                'good': (155, 175),       # 팔 거의 펴짐
                'ok': (140, 180),
                'bad': 'outside'
            },
            'body_tilt': {
                'good': (5, 15),          # 약간 앞으로
                'ok': (0, 20),
                'bad': 'outside'
            }
        }
    
    # ========================================
    # 공통 유틸리티 함수
    # ========================================
    
    def calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, 
                               p3: np.ndarray) -> float:
        """
        3점이 이루는 각도 계산 (p1-p2-p3)
        
        Args:
            p1, p2, p3: (x, y) 좌표 배열
        
        Returns:
            각도 (degree)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        return angle
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """두 점 사이 거리 계산 (정규화 좌표)"""
        return np.linalg.norm(p1 - p2)
    
    def get_person_height(self, keypoints: Dict) -> float:
        """
        사람 키 추정 (코 ~ 발목 거리의 평균)
        
        Args:
            keypoints: {'nose_y': 0.2, 'left_ankle_y': 0.8, ...}
        
        Returns:
            키 (정규화 좌표)
        """
        nose_y = keypoints['nose_y']
        left_ankle_y = keypoints['left_ankle_y']
        right_ankle_y = keypoints['right_ankle_y']
        
        # 두 발목 평균
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
        
        # 키 = 코 ~ 발목 거리
        height = avg_ankle_y - nose_y
        
        return height
    
    def get_shoulder_width(self, keypoints: Dict) -> float:
        """어깨 너비 계산"""
        left_shoulder = np.array([keypoints['left_shoulder_x'], 
                                 keypoints['left_shoulder_y']])
        right_shoulder = np.array([keypoints['right_shoulder_x'], 
                                  keypoints['right_shoulder_y']])
        
        return self.calculate_distance(left_shoulder, right_shoulder)
    
    # ========================================
    # 준비 자세 (KF1) 지표
    # ========================================
    
    def calc_foot_distance(self, keypoints: Dict) -> Tuple[float, str]:
        """
        1. 양발 거리 계산
        
        기준: 어깨 너비의 1~1.5배
        
        Returns:
            (거리비율, 진단)
        """
        # 양 발목 좌표
        left_ankle = np.array([keypoints['left_ankle_x'], 
                              keypoints['left_ankle_y']])
        right_ankle = np.array([keypoints['right_ankle_x'], 
                               keypoints['right_ankle_y']])
        
        # 발목 간 거리
        foot_distance = self.calculate_distance(left_ankle, right_ankle)
        
        # 어깨 너비
        shoulder_width = self.get_shoulder_width(keypoints)
        
        # 비율 계산
        ratio = foot_distance / shoulder_width if shoulder_width > 0 else 0
        
        # 진단
        good_range = self.thresholds['foot_distance']['good']
        ok_range = self.thresholds['foot_distance']['ok']
        
        if good_range[0] <= ratio <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= ratio <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return ratio, diagnosis
    
    def calc_racket_hand_height(self, keypoints: Dict, 
                                hand: str = 'right') -> Tuple[float, str]:
        """
        2. 라켓 손 높이 계산
        
        기준: 어깨보다 약간 위 (5~15cm)
        
        Args:
            hand: 'right' 또는 'left' (오른손잡이/왼손잡이)
        
        Returns:
            (높이차, 진단)
        """
        # 손목과 어깨 y 좌표
        wrist_y = keypoints[f'{hand}_wrist_y']
        shoulder_y = keypoints[f'{hand}_shoulder_y']
        
        # 높이 차이 (정규화 좌표, 위로 갈수록 작아짐)
        height_diff = shoulder_y - wrist_y  # 양수면 손목이 어깨보다 위
        
        # 진단
        good_range = self.thresholds['racket_hand_height']['good']
        ok_range = self.thresholds['racket_hand_height']['ok']
        
        if good_range[0] <= height_diff <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= height_diff <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return height_diff, diagnosis
    
    def calc_knee_angle(self, keypoints: Dict, side: str = 'right') -> Tuple[float, str]:
        """
        3. 무릎 각도 계산 (무게중심)
        
        기준: 150~170도 (약간 굽힘)
        
        Args:
            side: 'right' 또는 'left'
        
        Returns:
            (각도, 진단)
        """
        # 엉덩이-무릎-발목
        hip = np.array([keypoints[f'{side}_hip_x'], 
                       keypoints[f'{side}_hip_y']])
        knee = np.array([keypoints[f'{side}_knee_x'], 
                        keypoints[f'{side}_knee_y']])
        ankle = np.array([keypoints[f'{side}_ankle_x'], 
                         keypoints[f'{side}_ankle_y']])
        
        # 각도 계산
        angle = self.calculate_angle_3points(hip, knee, ankle)
        
        # 진단
        good_range = self.thresholds['knee_angle']['good']
        ok_range = self.thresholds['knee_angle']['ok']
        
        if good_range[0] <= angle <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= angle <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return angle, diagnosis
    
    # ========================================
    # 백스윙 (KF2) 지표
    # ========================================
    
    def calc_elbow_height(self, keypoints: Dict, 
                         hand: str = 'right') -> Tuple[float, str]:
        """
        4. 팔꿈치 높이 (최고점)
        
        기준: 어깨보다 15~25cm 위 (논문 기반)
        
        Returns:
            (높이차, 진단)
        """
        elbow_y = keypoints[f'{hand}_elbow_y']
        shoulder_y = keypoints[f'{hand}_shoulder_y']
        
        # 높이 차이 (양수면 팔꿈치가 어깨보다 위)
        height_diff = shoulder_y - elbow_y
        
        # 진단
        good_range = self.thresholds['elbow_height']['good']
        ok_range = self.thresholds['elbow_height']['ok']
        
        if good_range[0] <= height_diff <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= height_diff <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return height_diff, diagnosis
    
    def calc_shoulder_rotation(self, keypoints: Dict) -> Tuple[float, str]:
        """
        5. 어깨 회전각
        
        기준: 45~60도 (논문 기반)
        
        Returns:
            (회전각, 진단)
        """
        # 양 어깨 좌표
        left_shoulder = np.array([keypoints['left_shoulder_x'], 
                                 keypoints['left_shoulder_y']])
        right_shoulder = np.array([keypoints['right_shoulder_x'], 
                                  keypoints['right_shoulder_y']])
        
        # 어깨 벡터
        shoulder_vector = right_shoulder - left_shoulder
        
        # 정면 벡터 (y축)
        front_vector = np.array([0, 1])
        
        # 각도 계산
        cos_angle = np.dot(shoulder_vector, front_vector) / (
            np.linalg.norm(shoulder_vector) + 1e-6
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        # 90도 기준 회전각 (0도=정면, 90도=완전 옆)
        rotation = abs(90 - angle)
        
        # 진단
        good_range = self.thresholds['shoulder_rotation']['good']
        ok_range = self.thresholds['shoulder_rotation']['ok']
        
        if good_range[0] <= rotation <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= rotation <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return rotation, diagnosis
    
    def calc_left_hand_position(self, keypoints: Dict) -> Tuple[float, str]:
        """
        6. 왼손(비라켓손) 위치
        
        기준: 코보다 앞쪽 (셔틀콕 방향)
        
        Returns:
            (거리차, 진단)
        """
        # 왼손과 코의 x 좌표
        left_wrist_x = keypoints['left_wrist_x']
        nose_x = keypoints['nose_x']
        
        # x 좌표 차이 (양수면 왼손이 코보다 앞)
        x_diff = abs(left_wrist_x - nose_x)
        
        # 진단
        good_range = self.thresholds['left_hand_position']['good']
        ok_range = self.thresholds['left_hand_position']['ok']
        
        if good_range[0] <= x_diff <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= x_diff <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return x_diff, diagnosis
    
    # ========================================
    # 임팩트 (KF3) 지표
    # ========================================
    
    def calc_impact_height(self, keypoints: Dict, 
                          hand: str = 'right') -> Tuple[float, str]:
        """
        7. 타구 높이
        
        기준: 사용자 키의 85~95% 높이
        
        Returns:
            (키 대비 비율, 진단)
        """
        # 사람 키
        person_height = self.get_person_height(keypoints)
        
        # 타구 손목 높이
        wrist_y = keypoints[f'{hand}_wrist_y']
        nose_y = keypoints['nose_y']
        
        # 코 기준 손목 높이
        wrist_height = wrist_y - nose_y
        
        # 키 대비 비율
        height_ratio = wrist_height / person_height if person_height > 0 else 0
        
        # 진단
        good_range = self.thresholds['impact_height']['good']
        ok_range = self.thresholds['impact_height']['ok']
        
        if good_range[0] <= height_ratio <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= height_ratio <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return height_ratio, diagnosis
    
    def calc_arm_straightness(self, keypoints: Dict, 
                             hand: str = 'right') -> Tuple[float, str]:
        """
        8. 팔의 직진성
        
        기준: 155~175도 (거의 펴짐, 논문 160.5도)
        
        Returns:
            (각도, 진단)
        """
        # 어깨-팔꿈치-손목
        shoulder = np.array([keypoints[f'{hand}_shoulder_x'], 
                           keypoints[f'{hand}_shoulder_y']])
        elbow = np.array([keypoints[f'{hand}_elbow_x'], 
                        keypoints[f'{hand}_elbow_y']])
        wrist = np.array([keypoints[f'{hand}_wrist_x'], 
                        keypoints[f'{hand}_wrist_y']])
        
        # 각도 계산
        angle = self.calculate_angle_3points(shoulder, elbow, wrist)
        
        # 진단
        good_range = self.thresholds['arm_straightness']['good']
        ok_range = self.thresholds['arm_straightness']['ok']
        
        if good_range[0] <= angle <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= angle <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return angle, diagnosis
    
    def calc_body_tilt(self, keypoints: Dict) -> Tuple[float, str]:
        """
        9. 몸의 기울기
        
        기준: 약간 앞으로 (5~15도)
        
        Returns:
            (기울기 각도, 진단)
        """
        # 코와 엉덩이 중심
        nose = np.array([keypoints['nose_x'], keypoints['nose_y']])
        left_hip = np.array([keypoints['left_hip_x'], keypoints['left_hip_y']])
        right_hip = np.array([keypoints['right_hip_x'], keypoints['right_hip_y']])
        
        # 엉덩이 중심
        hip_center = (left_hip + right_hip) / 2
        
        # 코-엉덩이 벡터
        body_vector = nose - hip_center
        
        # 수직 벡터 (위쪽)
        vertical_vector = np.array([0, -1])  # y축 위쪽
        
        # 각도 계산
        cos_angle = np.dot(body_vector, vertical_vector) / (
            np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        # 진단
        good_range = self.thresholds['body_tilt']['good']
        ok_range = self.thresholds['body_tilt']['ok']
        
        if good_range[0] <= angle <= good_range[1]:
            diagnosis = '좋아요'
        elif ok_range[0] <= angle <= ok_range[1]:
            diagnosis = '아쉬워요'
        else:
            diagnosis = '나빠요'
        
        return angle, diagnosis
    
    # ========================================
    # 통합 분석 함수
    # ========================================
    
    def analyze_keyframe(self, keypoints: Dict, phase: str, 
                        hand: str = 'right') -> Dict:
        """
        Key Frame별 종합 분석
        
        Args:
            keypoints: Keypoint 딕셔너리
            phase: 'KF1', 'KF2', 'KF3'
            hand: 'right' 또는 'left'
        
        Returns:
            분석 결과 딕셔너리
        """
        results = {}
        
        if phase == 'KF1':
            # 준비 자세
            results['foot_distance'] = {
                'value': self.calc_foot_distance(keypoints)[0],
                'diagnosis': self.calc_foot_distance(keypoints)[1],
                'name': '양발 거리'
            }
            results['racket_hand_height'] = {
                'value': self.calc_racket_hand_height(keypoints, hand)[0],
                'diagnosis': self.calc_racket_hand_height(keypoints, hand)[1],
                'name': '라켓 손 높이'
            }
            results['knee_angle'] = {
                'value': self.calc_knee_angle(keypoints, hand)[0],
                'diagnosis': self.calc_knee_angle(keypoints, hand)[1],
                'name': '무릎 각도'
            }
        
        elif phase == 'KF2':
            # 백스윙
            results['elbow_height'] = {
                'value': self.calc_elbow_height(keypoints, hand)[0],
                'diagnosis': self.calc_elbow_height(keypoints, hand)[1],
                'name': '팔꿈치 높이'
            }
            results['shoulder_rotation'] = {
                'value': self.calc_shoulder_rotation(keypoints)[0],
                'diagnosis': self.calc_shoulder_rotation(keypoints)[1],
                'name': '어깨 회전각'
            }
            results['left_hand_position'] = {
                'value': self.calc_left_hand_position(keypoints)[0],
                'diagnosis': self.calc_left_hand_position(keypoints)[1],
                'name': '왼손 위치'
            }
        
        elif phase == 'KF3':
            # 임팩트
            results['impact_height'] = {
                'value': self.calc_impact_height(keypoints, hand)[0],
                'diagnosis': self.calc_impact_height(keypoints, hand)[1],
                'name': '타구 높이'
            }
            results['arm_straightness'] = {
                'value': self.calc_arm_straightness(keypoints, hand)[0],
                'diagnosis': self.calc_arm_straightness(keypoints, hand)[1],
                'name': '팔 직진성'
            }
            results['body_tilt'] = {
                'value': self.calc_body_tilt(keypoints)[0],
                'diagnosis': self.calc_body_tilt(keypoints)[1],
                'name': '몸 기울기'
            }
        
        return results
    
    def generate_feedback(self, results: Dict, phase: str) -> str:
        """
        분석 결과 기반 피드백 생성
        
        Args:
            results: analyze_keyframe() 결과
            phase: 'KF1', 'KF2', 'KF3'
        
        Returns:
            피드백 문자열
        """
        phase_names = {
            'KF1': '준비 자세',
            'KF2': '백스윙',
            'KF3': '임팩트'
        }
        
        feedback = f"\n{'='*50}\n"
        feedback += f"📊 {phase_names[phase]} 분석 결과\n"
        feedback += f"{'='*50}\n\n"
        
        for metric_key, metric_data in results.items():
            name = metric_data['name']
            value = metric_data['value']
            diagnosis = metric_data['diagnosis']
            
            # 이모지 선택
            emoji = '✅' if diagnosis == '좋아요' else '⚠️' if diagnosis == '아쉬워요' else '❌'
            
            # 값 포맷팅
            if 'angle' in metric_key or 'rotation' in metric_key or 'tilt' in metric_key:
                value_str = f"{value:.1f}°"
            elif 'ratio' in metric_key or 'height' in metric_key:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.2f}"
            
            feedback += f"{emoji} {name}: {value_str} - {diagnosis}\n"
        
        feedback += f"\n{'='*50}\n"
        
        return feedback


# ========================================
# 테스트 코드
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("배드민턴 스윙 6대 지표 계산 모듈 테스트")
    print("=" * 60)
    
    # 더미 데이터 (실제로는 CSV에서 읽어옴)
    dummy_keypoints = {
        # 얼굴
        'nose_x': 0.5, 'nose_y': 0.2,
        
        # 어깨
        'left_shoulder_x': 0.4, 'left_shoulder_y': 0.35,
        'right_shoulder_x': 0.6, 'right_shoulder_y': 0.35,
        
        # 팔꿈치
        'left_elbow_x': 0.3, 'left_elbow_y': 0.45,
        'right_elbow_x': 0.7, 'right_elbow_y': 0.25,
        
        # 손목
        'left_wrist_x': 0.25, 'left_wrist_y': 0.55,
        'right_wrist_x': 0.75, 'right_wrist_y': 0.15,
        
        # 엉덩이
        'left_hip_x': 0.42, 'left_hip_y': 0.6,
        'right_hip_x': 0.58, 'right_hip_y': 0.6,
        
        # 무릎
        'left_knee_x': 0.40, 'left_knee_y': 0.75,
        'right_knee_x': 0.60, 'right_knee_y': 0.75,
        
        # 발목
        'left_ankle_x': 0.38, 'left_ankle_y': 0.9,
        'right_ankle_x': 0.62, 'right_ankle_y': 0.9
    }
    
    # 분석기 생성
    analyzer = BadmintonMetrics()
    
    # 각 Phase 분석
    for phase in ['KF1', 'KF2', 'KF3']:
        results = analyzer.analyze_keyframe(dummy_keypoints, phase)
        feedback = analyzer.generate_feedback(results, phase)
        print(feedback)