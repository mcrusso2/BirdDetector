import argparse
import csv
import random
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

#
# ---------------------- CONFIG ----------------------
#

# Valid Open Images bird classes grouped into subclasses
BIRD_SUBCLASSES = {
    # Small birds
    "Sparrow": "small_bird",
    "Finch": "small_bird",

    # Raptors
    "Eagle": "raptor",
    "Hawk": "raptor",
    "Falcon": "raptor",
    "Owl": "raptor",
    "Bird of prey": "raptor",

    # Waterfowl
    "Duck": "waterfowl",
    "Goose": "waterfowl",
    "Swan": "waterfowl",
    "Water bird": "waterfowl",

    # Corvids
    "Crow": "corvid",
    "Raven": "corvid",

    # Other distinct groups
    "Parrot": "parrot",
    "Penguin": "penguin",
    "Woodpecker": "woodpecker",
    "Seagull": "seagull",

    # Generic fallback
    "Bird": "generic_bird",
}

PRIMARY_ADVERSARY_CLASSES = {
    "Squirrel": "no_bird",
    "Cat": "no_bird",
    "Dog": "no_bird",
    "Raccoon": "no_bird",
    "Rodent": "no_bird",
    "Person": "no_bird",
}

FEEDER_CONTEXT_CLASSES = {
    "Bird feeder": "no_bird",
}

AUX_NEGATIVE_CLASSES = {
    "Tree": "no_bird",
    "House": "no_bird",
    "Window": "no_bird",
    "Fence": "no_bird",
}

# Only valid Open Images classes included
PRESETS = {
    "bird_subclasses": {
        # Small birds
        "Sparrow": 2000,
        "Finch": 2000,

        # Raptors
        "Eagle": 1500,
        "Hawk": 1500,
        "Falcon": 1500,
        "Owl": 1500,
        "Bird of prey": 1500,

        # Waterfowl
        "Duck": 2000,
        "Goose": 2000,
        "Swan": 2000,
        "Water bird": 2000,

        # Corvids
        "Crow": 1500,
        "Raven": 1500,

        # Other bird groups
        "Parrot": 1500,
        "Penguin": 1500,
        "Woodpecker": 1500,
        "Seagull": 1500,

        # Generic bird
        "Bird": 3000,

        # Negatives
        "Squirrel": 4000,
        "Cat": 3000,
        "Dog": 3000,
        "Raccoon": 2000,
        "Rodent": 2000,
        "Person": 4000,
        "Bird feeder": 4000,
        "Tree": 3000,
        "House": 3000,
        "Window": 2000,
        "Fence": 2000,
    }
}

#
# ---------------------- UTILS ----------------------
#

def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def assign_split(idx: int, total: int, args):
    r = random.random()
    if r < args.train_ratio:
        return "train"
    elif r < args.train_ratio + args.val_ratio:
        return "val"
    else:
        return "test"


#
# ---------------------- LOADING ----------------------
#

def load_openimages_class(class_name: str, split: str, max_samples: int):
    dataset_name = f"oi7_det_{split}_{safe_name(class_name)}_{max_samples}"

    print("\nLoading Open Images V7 (detection labels)")
    print(f"  class:       {class_name}")
    print(f"  split:       {split}")
    print(f"  max_samples: {max_samples}")

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=[class_name],
            max_samples=max_samples,
            shuffle=True,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        print(f"WARNING: failed to load detection class {class_name!r}. Skipping.")
        print(f"Reason: {exc}")
        return None

    print(dataset)
    return dataset


def load_openimages_imagelevel(class_name: str, split: str, max_samples: int):
    dataset_name = f"oi7_imagelevel_{split}_{safe_name(class_name)}_{max_samples}"

    print("\nLoading Open Images V7 (image-level labels)")
    print(f"  class:       {class_name}")
    print(f"  split:       {split}")
    print(f"  max_samples: {max_samples}")

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["classifications"],
            classes=[class_name],
            max_samples=max_samples,
            shuffle=True,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        print(f"WARNING: failed to load image-level class {class_name!r}. Skipping.")
        print(f"Reason: {exc}")
        return None

    print(dataset)
    return dataset


#
# ---------------------- EXPORT ----------------------
#

def save_crop(sample, det, classifier_label, split, output_dir: Path, records, args):
    img = sample.filepath
    w = sample.metadata.width
    h = sample.metadata.height

    x, y, bw, bh = det.bounding_box
    abs_w = bw * w
    abs_h = bh * h
    box_area = (abs_w * abs_h) / (w * h)

    if box_area < args.min_box_area:
        return False

    pad_x = args.crop_pad * abs_w
    pad_y = args.crop_pad * abs_h

    x1 = max(0, int((x * w) - pad_x))
    y1 = max(0, int((y * h) - pad_y))
    x2 = min(w, int((x * w + abs_w) + pad_x))
    y2 = min(h, int((y * h + abs_h) + pad_y))

    if x2 <= x1 or y2 <= y1:
        return False

    from PIL import Image
    im = Image.open(img).convert("RGB")
    crop = im.crop((x1, y1, x2, y2))

    out_dir = output_dir / "crops" / split / classifier_label
    ensure_dir(out_dir)

    out_path = out_dir / f"{sample.id}_{det.label}_{random.randint(0, 999999)}.jpg"
    crop.save(out_path)

    records.append(
        {
            "filepath": str(out_path),
            "split": split,
            "label": classifier_label,
            "source": "crop",
            "orig_path": img,
        }
    )
    return True


def copy_whole_frame(sample, classifier_label, split, output_dir: Path, records):
    src = sample.filepath
    out_dir = output_dir / "whole_frames" / split / classifier_label
    ensure_dir(out_dir)

    out_path = out_dir / f"{sample.id}_{random.randint(0, 999999)}.jpg"
    shutil.copy(src, out_path)

    records.append(
        {
            "filepath": str(out_path),
            "split": split,
            "label": classifier_label,
            "source": "whole",
            "orig_path": src,
        }
    )


def export_dataset_for_class(dataset, class_name, args, records):
    # Updated classifier label logic
    if class_name in BIRD_SUBCLASSES:
        classifier_label = BIRD_SUBCLASSES[class_name]
    elif class_name in PRIMARY_ADVERSARY_CLASSES:
        classifier_label = PRIMARY_ADVERSARY_CLASSES[class_name]
    elif class_name in FEEDER_CONTEXT_CLASSES:
        classifier_label = FEEDER_CONTEXT_CLASSES[class_name]
    elif class_name in AUX_NEGATIVE_CLASSES:
        classifier_label = AUX_NEGATIVE_CLASSES[class_name]
    else:
        classifier_label = "no_bird"

    print(f"\nExporting class {class_name!r} as {classifier_label!r}")

    for sample in dataset:
        # Detect whether this sample has detections
        if hasattr(sample, "detections") and sample.detections is not None:
            dets = sample.detections.detections
        else:
            dets = []

        # 1) Export crops (only for detection datasets)
        crops_saved = 0
        if dets:
            for det in dets:
                if det.label != class_name:
                    continue
                split = assign_split(0, 1, args)
                if save_crop(sample, det, classifier_label, split, args.output_dir, records, args):
                    crops_saved += 1
                if crops_saved >= args.max_crops_per_image:
                    break

        # 2) Export whole frame
        if args.export_mode in ("whole", "both"):
            split = assign_split(0, 1, args)
            copy_whole_frame(sample, classifier_label, split, args.output_dir, records)


def print_summary(records):
    print("\nFinal summary:")
    counts = {}
    for r in records:
        key = (r["source"], r["split"], r["label"])
        counts[key] = counts.get(key, 0) + 1

    for (src, split, label), n in sorted(counts.items()):
        print(f"  {src:5s} | {split:5s} | {label:11s} : {n}")


#
# ---------------------- MAIN ----------------------
#

def main(args):
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "manifests")

    args.output_dir = output_dir

    targets = PRESETS[args.preset]

    records = []
    datasets = []

    for class_name, max_samples in targets.items():
        # 1) Detection-based dataset
        dataset_det = load_openimages_class(
            class_name=class_name,
            split=args.openimages_split,
            max_samples=max_samples,
        )

        # 2) Image-level dataset (only for valid bird subclasses)
        dataset_img = None
        if class_name in BIRD_SUBCLASSES:
            dataset_img = load_openimages_imagelevel(
                class_name=class_name,
                split=args.openimages_split,
                max_samples=max_samples,
            )

        # Export detection-based samples
        if dataset_det is not None:
            datasets.append(dataset_det)
            export_dataset_for_class(
                dataset=dataset_det,
                class_name=class_name,
                args=args,
                records=records,
            )

        # Export image-level samples as whole frames
        if dataset_img is not None:
            datasets.append(dataset_img)
            classifier_label = BIRD_SUBCLASSES[class_name]
            print(f"\nExporting image-level class {class_name!r} as {classifier_label!r}")
            for sample in dataset_img:
                split = assign_split(0, 1, args)
                copy_whole_frame(
                    sample=sample,
                    classifier_label=classifier_label,
                    split=split,
                    output_dir=output_dir,
                    records=records,
                )

    # Write manifest
    manifest_path = output_dir / "manifests" / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "split", "label", "source", "orig_path"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print_summary(records)

    if args.view:
        merged = fo.Dataset()
        for ds in datasets:
            merged.merge_samples(ds)
        session = fo.launch_app(merged)
        session.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./data/openimages_bird_subclasses")
    parser.add_argument("--openimages_split", type=str, default="train")
    parser.add_argument("--preset", type=str, default="bird_subclasses")
    parser.add_argument("--export-mode", type=str, default="both")
    parser.add_argument("--min-box-area", type=float, default=0.001)
    parser.add_argument("--max-crops-per-image", type=int, default=5)
    parser.add_argument("--crop-pad", type=float, default=0.25)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view", action="store_true")

    args = parser.parse_args()
    main(args)
