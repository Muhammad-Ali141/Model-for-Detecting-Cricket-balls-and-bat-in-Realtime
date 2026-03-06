# Model for Detecting Cricket Balls and Bats in Realtime

YOLOv8-based object detection for real-time cricket **ball** and **bat** detection in videos. Custom-trained on separate datasets; the notebook runs both models and draws ball (circle/box) and bat (rectangle) on each frame.

---

## Features

- **Ball model** (`ball_best.pt`) — cricket ball detection with temporal consistency
- **Bat model** (`bat_best.pt`) — cricket bat detection
- **COCO → YOLO** dataset conversion and train/val splits
- **Training scripts**: `train_ball_model.py` (train_ball/), `train_bat_model.py` (train_bat/, merges train/ then deletes it)
- **Inference notebook** (`Ball_Detection.ipynb`) — processes videos from `labelled/` and `labelled 2/`
- **Output**: Annotated videos in `detection_output/`

---

## Requirements

```
pip install opencv-python numpy ultralytics matplotlib
```

- Python 3.8+
- PyTorch (installed via `ultralytics`)

---

## Datasets

- **Ball**: COCO-annotated images in `train_ball/` — single class `ball`. ~7k images, 80/20 split.
- **Bat**: COCO-annotated images in `train_bat/`; `train_bat_model.py` can merge in `train/` then delete it. ~2.8k images after merge, 80/20 split.

---

## Training

**Ball** — dataset in `train_ball/`:
```bash
python train_ball_model.py
```
→ `ball_dataset/`, `runs/ball_train/`, `ball_best.pt`

**Bat** — dataset in `train_bat/` (optionally merge `train/` into it, then script deletes `train/`):
```bash
python train_bat_model.py
```
→ `bat_dataset/`, `runs/bat_train/`, `bat_best.pt`

Both use YOLOv8n, image size 640, batch 16, seed 42. Ball: 80/20 split; bat: 80/20 after merge.

---

## Model Performance

### Ball (runs/ball_train)

| Metric       | Value   |
|-------------|---------|
| Precision   | 89.01%  |
| Recall      | 81.64%  |
| mAP@50      | 87.04%  |
| mAP@50-95   | 51.40%  |

### Bat (runs/bat_train)

| Metric       | Value   |
|-------------|---------|
| Precision   | 88.8%   |
| Recall      | 82.5%   |
| mAP@50      | 85.7%   |
| mAP@50-95   | 58.6%   |
| Val images  | 86      |
| Epochs      | 250     |

---

## Inference (Detection on Videos)

1. Open `Ball_Detection.ipynb` in Jupyter.
2. Ensure `ball_best.pt` (and optionally `bat_best.pt`) exist.
3. Place videos in `labelled/` and `labelled 2/`.
4. Run all cells. Outputs go to `detection_output/`.

The notebook runs both models, draws **ball** (circle + box) and **bat** (orange box), and previews with matplotlib.

---

## Project Structure

```
.
├── README.md
├── train_ball_model.py      # Ball training (train_ball/)
├── train_bat_model.py       # Bat training (train_bat/, merges train/ then deletes)
├── Ball_Detection.ipynb     # Ball + bat detection
├── ball_best.pt, bat_best.pt
├── ball_dataset/, bat_dataset/
├── runs/ball_train/, runs/bat_train/
├── train_ball/, train_bat/  # COCO source data
├── labelled/, labelled 2/
└── detection_output/
```

---

## Usage Summary

1. **Train ball**: `python train_ball_model.py` → `ball_best.pt`
2. **Train bat**: `python train_bat_model.py` → `bat_best.pt` (merges train/ into train_bat/, deletes train/)
3. **Detect**: Run `Ball_Detection.ipynb` → `*_detected.mp4` in `detection_output/`

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) — Cricket Dataset
- OpenCV, PyTorch
