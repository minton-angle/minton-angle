import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
from datetime import datetime
import os
from pathlib import Path

# 모델 로드
model = YOLO("app/services/grip/grip_classification_yolo11n_best.pt")

class GripService:
    @staticmethod
    def predict_grip(image_bytes):
        # 1. 저장 경로 설정
        # 프로젝트 루트 기준 backend/data/grip_test
        save_dir = Path("data/grip_test")
        save_dir.mkdir(parents=True, exist_ok=True)

        # 2. 파일 번호 매기기 (기존 파일 개수 확인)
        existing_files = list(save_dir.glob("*.jpg"))
        file_count = len(existing_files) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_count}_{timestamp}.jpg"
        save_path = str(save_dir / filename)

        # 3. 이미지 변환 (PIL -> OpenCV BGR)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 4. YOLO 추론
        results = model.predict(img_cv, conf=0.1) # 탐지율을 위해 conf 하향
        result = results[0]

        # 5. 결과 그리기 (OpenCV Overlay)
        feedback_map = {
            0: ("CORRECT", (0, 255, 0)),    # 초록
            1: ("INDEX_UP", (0, 0, 255)),   # 빨강
            2: ("ORDER_ERR", (0, 0, 255)),
            3: ("TENNIS", (0, 0, 255)),
            4: ("THUMB_UP", (0, 0, 255)),
            5: ("FAIL", (128, 128, 128))    # 회색
        }

        class_id = 5
        confidence = 0.0
        box_coords = []

        if len(result.boxes) > 0:
            box = result.boxes[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            box_coords = box.xyxy[0].tolist()

            # 이미지 위에 박스와 라벨 그리기
            x1, y1, x2, y2 = map(int, box_coords)
            label_text, color = feedback_map.get(class_id, feedback_map[5])
            
            # 사각형 그리기
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
            # 라벨 텍스트 배경
            cv2.rectangle(img_cv, (x1, y1 - 35), (x1 + 200, y1), color, -1)
            # 라벨 텍스트
            cv2.putText(img_cv, f"{label_text} {confidence:.2f}", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 6. 로컬 폴더에 이미지 저장
        cv2.imwrite(save_path, img_cv)
        print(f"📸 분석 결과 저장 완료: {save_path}")

        return {
            "class_id": class_id,
            "confidence": confidence,
            "box": box_coords,
            "save_path": save_path  # 저장된 경로도 반환 (필요시)
        }