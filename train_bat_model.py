"""
Train cricket bat detection model from COCO-annotated dataset.
Combines train_bat/ and train/ (bat annotations) into train_bat/, then converts
to YOLO format, splits train/val, and trains. Deletes train/ after merge.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_bat"
TRAIN_EXTRA_DIR = BASE_DIR / "train"  # merged into train_bat then deleted
COCO_JSON = TRAIN_DIR / "_annotations.coco.json"
DATASET_DIR = BASE_DIR / "bat_dataset"
OUTPUT_DIR = BASE_DIR / "runs" / "bat_train"
BEST_MODEL_PATH = BASE_DIR / "bat_best.pt"

# Training config for ~2.8k images after merging train_bat + train
EPOCHS = 250
IMGSZ = 640
BATCH = 16
VAL_SPLIT = 0.2    # 80% train / 20% val (~2240 train, ~560 val)
SEED = 42


def merge_train_into_train_bat():
    """
    Merge train/ (COCO bat data) into train_bat/: combine annotations and copy
    images with unique names, then delete train/.
    """
    if not TRAIN_EXTRA_DIR.exists():
        return
    extra_coco_path = TRAIN_EXTRA_DIR / "_annotations.coco.json"
    if not extra_coco_path.exists():
        shutil.rmtree(TRAIN_EXTRA_DIR, ignore_errors=True)
        return

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(COCO_JSON, encoding="utf-8") as f:
        coco_bat = json.load(f)
    with open(extra_coco_path, encoding="utf-8") as f:
        coco_extra = json.load(f)

    # Merge images: keep train_bat as-is; add train with new ids and prefixed filenames
    images_bat = coco_bat.get("images", [])
    images_extra = coco_extra.get("images", [])
    max_id = max((img["id"] for img in images_bat), default=0)
    old_to_new_id = {}
    prefix = "extra_"
    merged_images = list(images_bat)
    for img in images_extra:
        max_id += 1
        old_to_new_id[img["id"]] = max_id
        new_fname = prefix + img["file_name"]
        merged_images.append({
            "id": max_id,
            "file_name": new_fname,
            "width": img["width"],
            "height": img["height"],
        })

    # Merge annotations: train_bat as-is (category -> 0); train remap image_id, category_id -> 0
    anns_bat = []
    for i, a in enumerate(coco_bat.get("annotations", [])):
        anns_bat.append({
            "id": a.get("id", i + 1),
            "image_id": a["image_id"],
            "category_id": 0,
            "bbox": a["bbox"],
        })
    next_ann_id = max((a["id"] for a in anns_bat), default=0) + 1
    merged_anns = list(anns_bat)
    for a in coco_extra.get("annotations", []):
        if a["image_id"] not in old_to_new_id:
            continue
        merged_anns.append({
            "id": next_ann_id,
            "image_id": old_to_new_id[a["image_id"]],
            "category_id": 0,
            "bbox": a["bbox"],
        })
        next_ann_id += 1

    merged = {
        "images": merged_images,
        "annotations": merged_anns,
        "categories": [{"id": 0, "name": "bat"}],
    }
    with open(COCO_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    # Copy images from train/ to train_bat/ with new names
    for img in images_extra:
        src = TRAIN_EXTRA_DIR / img["file_name"]
        if not src.exists():
            continue
        dst = TRAIN_DIR / (prefix + img["file_name"])
        shutil.copy2(src, dst)

    shutil.rmtree(TRAIN_EXTRA_DIR, ignore_errors=True)
    print(f"  Merged train/ into train_bat/ and removed train/. Total images: {len(merged_images)}")


def convert_coco_to_yolo(coco_path: Path, images_dir: Path, labels_dir: Path):
    """Convert COCO annotations to YOLO format. Single class: bat (all categories -> 0)."""
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_img = {}
    for img in coco["images"]:
        id_to_img[img["id"]] = (img["width"], img["height"], img["file_name"])

    # All categories in COCO -> single class 0 (bat)
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
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

    all_labels = DATASET_DIR / "labels" / "all"
    all_labels.mkdir(parents=True, exist_ok=True)
    n = convert_coco_to_yolo(COCO_JSON, TRAIN_DIR, all_labels)

    label_stems = {p.stem for p in all_labels.glob("*.txt")}
    image_files = [f for f in TRAIN_DIR.glob("*.jpg") if f.stem in label_stems]
    if not image_files:
        image_files = [f for f in TRAIN_DIR.glob("*.jpeg") if f.stem in label_stems]
    if not image_files:
        image_files = [f for f in TRAIN_DIR.glob("*.png") if f.stem in label_stems]
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

    shutil.rmtree(all_labels, ignore_errors=True)
    return len(train_files), len(val_files)


def create_data_yaml():
    data_yaml = DATASET_DIR / "data.yaml"
    path = str(DATASET_DIR.resolve()).replace("\\", "/")
    content = f"""# Cricket bat detection dataset
path: {path}
train: images/train
val: images/val

nc: 1
names: ['bat']
"""
    data_yaml.write_text(content)
    return data_yaml


def train():
    from ultralytics import YOLO

    print("Merging train/ into train_bat/ (if train/ exists)...")
    merge_train_into_train_bat()

    print("Preparing dataset...")
    n_train, n_val = prepare_dataset()
    print(f"  Train: {n_train}, Val: {n_val}")

    data_yaml = create_data_yaml()
    print(f"  data.yaml: {data_yaml}")

    model = YOLO("yolov8n.pt")
    print("\nTraining YOLOv8n on cricket bat dataset...")
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(OUTPUT_DIR.parent),
        name="bat_train",
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
