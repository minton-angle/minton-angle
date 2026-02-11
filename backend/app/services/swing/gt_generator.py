"""
GT Generator - 전문가 영상에서 기준 범위 추출

사용법:
    python gt_generator.py --input ../../data/expert_videos/
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from engine.pose_extractor import PoseExtractor
from engine.swing_analyzer import SwingAnalyzer
from engine.constants import OUTPUT_FILES, EXPERT_FILES, PHASE1, PHASE2, PHASE3


# ============================================
# 경로 설정
# ============================================
def get_paths():
    """프로젝트 경로들"""
    current = Path(__file__).resolve().parent
    
    # backend/app/services/swing/ 기준
    backend = current.parents[2]  # backend/
    project = backend.parent       # project root
    frontend = project / "frontend"
    
    return {
        "expert_videos": backend / "data" / "expert_videos",
        "expert_gt": backend / "data" / "expert_gt", 
        "criteria": backend / "data" / "swing_criteria.json",
        "frontend_images": frontend / "assets" / "images",
        "frontend_videos": frontend / "assets" / "videos",
    }


# ============================================
# 영상 클립 추출
# ============================================
def save_video_clip(video_path: str, start: int, end: int, output_path: str) -> bool:
    """영상 클립 저장"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        return True
    except:
        return False


# ============================================
# GT Generator
# ============================================
class GTGenerator:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.paths = get_paths()
        self.extractor = PoseExtractor()
        self.analyzer = SwingAnalyzer()
        self.all_metrics = []
    
    def process_video(self, video_path: Path) -> Dict:
        """단일 영상 처리"""
        key_frames = self.extractor.extract_from_video(str(video_path))
        result = self.analyzer.analyze(key_frames)
        
        # 원본 값만 추출 (점수 제외)
        metrics = {
            "video": video_path.name,
            "frames": {
                "ready": key_frames.ready.frame_idx,
                "impact": key_frames.impact.frame_idx,
                "end": key_frames.end.frame_idx
            },
            PHASE1: {k: v["value"] for k, v in result["phases"][PHASE1]["metrics"].items()},
            PHASE2: {k: v["value"] for k, v in result["phases"][PHASE2]["metrics"].items()},
            PHASE3: {k: v["value"] for k, v in result["phases"][PHASE3]["metrics"].items()},
        }
        
        return metrics, key_frames
    
    def save_expert_assets(self, video_path: Path, key_frames):
        """대표 전문가 에셋 저장"""
        paths = self.paths
        paths["frontend_images"].mkdir(parents=True, exist_ok=True)
        paths["frontend_videos"].mkdir(parents=True, exist_ok=True)
        
        # 이미지
        cv2.imwrite(str(paths["frontend_images"] / EXPERT_FILES["ready_image"]), 
                    key_frames.ready.image)
        cv2.imwrite(str(paths["frontend_images"] / EXPERT_FILES["impact_image"]), 
                    key_frames.impact.image)
        
        # 영상 클립
        save_video_clip(str(video_path), 
                       key_frames.ready.frame_idx, 
                       key_frames.impact.frame_idx,
                       str(paths["frontend_videos"] / EXPERT_FILES["swing_video"]))
        
        save_video_clip(str(video_path),
                       key_frames.impact.frame_idx,
                       key_frames.end.frame_idx,
                       str(paths["frontend_videos"] / EXPERT_FILES["followthrough_video"]))
        
        print("  → 전문가 에셋 저장 완료")
    
    def run(self):
        """전체 실행"""
        paths = self.paths
        paths["expert_gt"].mkdir(parents=True, exist_ok=True)
        
        # 영상 찾기
        videos = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.MP4"))
        videos = sorted(videos)
        
        if not videos:
            print(f"⚠️ 영상 없음: {self.input_dir}")
            return
        
        print(f"\n📁 {len(videos)}개 영상 처리")
        print("=" * 40)
        
        for i, vpath in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {vpath.name}")
            
            try:
                metrics, key_frames = self.process_video(vpath)
                self.all_metrics.append(metrics)
                
                # 개별 저장
                out_file = paths["expert_gt"] / f"{vpath.stem}.json"
                with open(out_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                print(f"  ✅ 저장: {out_file.name}")
                
                # 첫 번째 영상 = 대표 에셋
                if i == 1:
                    self.save_expert_assets(vpath, key_frames)
                    
            except Exception as e:
                print(f"  ❌ 오류: {e}")
        
        self.extractor.close()
        self.generate_criteria()
    
    def generate_criteria(self):
        """기준 범위 생성"""
        if not self.all_metrics:
            return
        
        def get_range(values):
            valid = [v for v in values if v is not None]
            if not valid:
                return {"min": None, "max": None}
            return {
                "min": round(min(valid), 4),
                "max": round(max(valid), 4),
                "mean": round(float(np.mean(valid)), 4)
            }
        
        # 값 수집
        p1 = {"ready_triangle": [], "left_hand_height": [], "chest_direction": []}
        p2 = {"elbow_angle": [], "impact_height": [], "hip_rotation": []}
        p3 = {"followthrough": []}
        
        for m in self.all_metrics:
            for k in p1: p1[k].append(m[PHASE1].get(k))
            for k in p2: p2[k].append(m[PHASE2].get(k))
            for k in p3: p3[k].append(m[PHASE3].get(k))
        
        criteria = {
            "generated_at": datetime.now().isoformat(),
            "expert_count": len(self.all_metrics),
            "criteria": {
                PHASE1: {
                    "ready_triangle": {
                        "type": "boolean",
                        "expected": True,
                        "true_count": sum(1 for v in p1["ready_triangle"] if v)
                    },
                    "left_hand_height": {"type": "range", **get_range(p1["left_hand_height"])},
                    "chest_direction": {"type": "range", **get_range(p1["chest_direction"])},
                },
                PHASE2: {
                    "elbow_angle": {"type": "range", **get_range(p2["elbow_angle"])},
                    "impact_height": {"type": "range", **get_range(p2["impact_height"])},
                    "hip_rotation": {"type": "range", **get_range(p2["hip_rotation"])},
                },
                PHASE3: {
                    "followthrough": {"type": "range", **get_range(p3["followthrough"])},
                }
            }
        }
        
        with open(self.paths["criteria"], 'w') as f:
            json.dump(criteria, f, indent=2)
        
        print(f"\n📊 Criteria 저장: {self.paths['criteria']}")
        print("\n=== 기준 범위 ===")
        for phase, metrics in criteria["criteria"].items():
            print(f"\n{phase}:")
            for name, data in metrics.items():
                if data.get("type") == "boolean":
                    print(f"  {name}: true_count={data.get('true_count')}")
                else:
                    print(f"  {name}: {data.get('min')} ~ {data.get('max')}")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="../../data/expert_videos/")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"폴더 생성됨: {input_dir}")
        print("전문가 영상을 넣고 다시 실행하세요.")
    else:
        print("🏸 GT Generator")
        GTGenerator(input_dir).run()
        print("\n✅ 완료!")