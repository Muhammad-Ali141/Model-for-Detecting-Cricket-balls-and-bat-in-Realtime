"""
Train cricket ball detection model from COCO-annotated dataset.
Converts train/_annotations.coco.json + images to YOLO format, splits train/val,
trains YOLOv8, and saves best.pt.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
COCO_JSON = TRAIN_DIR / "_annotations.coco.json"
DATASET_DIR = BASE_DIR / "ball_dataset"
OUTPUT_DIR = BASE_DIR / "runs" / "ball_train"
BEST_MODEL_PATH = BASE_DIR / "ball_best.pt"

# Training config
EPOCHS = 50
IMGSZ = 640
BATCH = 16
VAL_SPLIT = 0.2
SEED = 42


def convert_coco_to_yolo(coco_path: Path, images_dir: Path, labels_dir: Path):
    """Convert COCO annotations to YOLO format. Single class: ball (id 0)."""
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id -> (width, height, file_name)
    id_to_img = {}
    for img in coco["images"]:
        id_to_img[img["id"]] = (img["width"], img["height"], img["file_name"])

    # category: ball (0,1) -> 0, stump (2) -> skip for ball-only training
    BALL_IDS = {0, 1}
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in BALL_IDS:
            continue
        img_to_anns[ann["image_id"]].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for img_id, anns in img_to_anns.items():
        if img_id not in id_to_img:
            continue
        w, h, fname = id_to_img[img_id]
        stem = Path(fname).stem
        label_path = labels_dir / f"{stem}.txt"
        lines = []
        for a in anns:
            bbox = a["bbox"]
            x, y = float(bbox[0]), float(bbox[1])
            bw, bh = float(bbox[2]), float(bbox[3])
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if 0 <= cx <= 1 and 0 <= cy <= 1 and nw > 0 and nh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        if lines:
            label_path.write_text("".join(lines))
            converted += 1
    return converted


def prepare_dataset():
    """Convert COCO -> YOLO, split into train/val."""
    import random
    random.seed(SEED)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # Convert all to labels
    all_labels = DATASET_DIR / "labels" / "all"
    all_labels.mkdir(parents=True, exist_ok=True)
    n = convert_coco_to_yolo(COCO_JSON, TRAIN_DIR, all_labels)

    # Get image stems that have labels
    label_stems = {p.stem for p in all_labels.glob("*.txt")}
    image_files = [f for f in TRAIN_DIR.glob("*.jpg") if f.stem in label_stems]
    random.shuffle(image_files)

    n_val = int(len(image_files) * VAL_SPLIT)
    val_files = image_files[:n_val]
    train_files = image_files[n_val:]

    for f in train_files:
        shutil.copy(f, images_dir / "train" / f.name)
        lbl = all_labels / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, labels_dir / "train" / f"{f.stem}.txt")
    for f in val_files:
        shutil.copy(f, images_dir / "val" / f.name)
        lbl = all_labels / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, labels_dir / "val" / f"{f.stem}.txt")

    # Cleanup temp
    shutil.rmtree(all_labels, ignore_errors=True)
    return len(train_files), len(val_files)


def create_data_yaml():
    data_yaml = DATASET_DIR / "data.yaml"
    path = str(DATASET_DIR.resolve()).replace("\\", "/")
    content = f"""# Cricket ball detection dataset
path: {path}
train: images/train
val: images/val

nc: 1
names: ['ball']
"""
    data_yaml.write_text(content)
    return data_yaml


def train():
    from ultralytics import YOLO

    print("Preparing dataset...")
    n_train, n_val = prepare_dataset()
    print(f"  Train: {n_train}, Val: {n_val}")

    data_yaml = create_data_yaml()
    print(f"  data.yaml: {data_yaml}")

    model = YOLO("yolov8n.pt")
    print("\nTraining YOLOv8n on cricket ball dataset...")
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(OUTPUT_DIR.parent),
        name="ball_train",
        exist_ok=True,
        seed=SEED,
    )

    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, BEST_MODEL_PATH)
        print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    else:
        print(f"\nBest weights: {best_pt}")


if __name__ == "__main__":
    train()
