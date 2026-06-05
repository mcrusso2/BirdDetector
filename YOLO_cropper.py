# ---------------- generate_yolo_crops.py ----------------
#
# Detect birds in whole_frames using YOLOv8 and generate crops.
#

import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import argparse
import shutil

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def crop_and_save(img_path, det, out_dir):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # YOLO box format: x_center, y_center, width, height (all normalized)
    xc, yc, bw, bh = det.xywhn[0].tolist()

    x1 = int((xc - bw/2) * w)
    y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w)
    y2 = int((yc + bh/2) * h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    crop = img.crop((x1, y1, x2, y2))

    # FIX: convert tensors to Python types
    cls_id = int(det.cls)
    conf = float(det.conf)

    out_path = out_dir / f"{Path(img_path).stem}_{cls_id}_{conf:.2f}.jpg"
    crop.save(out_path)
    return True



def process_split(split, whole_root, crops_root, model):
    print(f"\nProcessing split: {split}")

    whole_dir = Path(whole_root) / split / "bird_present"
    out_dir = Path(crops_root) / split / "bird_present"
    ensure_dir(out_dir)

    count = 0

    for img_file in whole_dir.glob("*.jpg"):
        results = model(img_file)

        for det in results[0].boxes:
            cls = int(det.cls)
            if cls == 14:  # COCO class 14 = "bird"
                if crop_and_save(img_file, det, out_dir):
                    count += 1

    print(f"Saved {count} bird crops for split '{split}'")


def main(args):
    whole_root = Path(args.data_path) / "whole_frames"
    crops_root = Path(args.data_path) / "crops"

    # Load YOLOv8 pretrained on COCO
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8x.pt")  # best accuracy

    for split in ["train", "val", "test"]:
        process_split(split, whole_root, crops_root, model)

    print("\nYOLO crop generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default="./data/openimages_feeder_optimal")
    args = parser.parse_args()

    main(args)
