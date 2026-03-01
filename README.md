# Model for Detecting Cricket Balls in Realtime

A YOLOv8-based object detection model for real-time cricket ball detection in videos. Trained on a custom dataset of cricket ball images, the model detects the ball in video frames and draws bounding boxes for visualization.

---

## Features

- **Custom-trained YOLOv8n** model (`ball_best.pt`) for cricket ball detection
- **COCO → YOLO** dataset conversion and train/val split
- **Training pipeline** (`train_ball_model.py`) with reproducible settings
- **Inference notebook** (`Ball_Detection.ipynb`) for processing videos from `labelled/` and `labelled 2/`
- **Temporal consistency**: Uses previous frame ball position to disambiguate multiple detections
- **Output**: Annotated videos saved to `detection_output/`

---

## Requirements

```
pip install opencv-python numpy ultralytics matplotlib
```

- Python 3.8+
- PyTorch (installed via `ultralytics`)

---

## Dataset

- **Source**: COCO-annotated cricket ball images in `train/` with `_annotations.coco.json`
- **Format**: Single class — `ball` (stump annotations ignored)
- **Split**: 80% train, 20% validation (random seed 42)
- **Images**: ~6,996 annotated images (Cricket Dataset, Roboflow)

---

## Training

1. Place your COCO dataset in `train/`:
   - `train/_annotations.coco.json`
   - `train/*.jpg`

2. Run training:
   ```bash
   python train_ball_model.py
   ```

3. Outputs:
   - `ball_dataset/` — YOLO-format dataset
   - `runs/ball_train/` — training logs, metrics, and checkpoints
   - `ball_best.pt` — best model saved to project root

### Training Configuration

| Parameter   | Value |
|------------|-------|
| Model      | YOLOv8n (nano) |
| Epochs     | 50 |
| Image size | 640 |
| Batch size | 16 |
| Val split  | 0.2 |
| Seed       | 42 |

---

## Model Performance

Results from `runs/ball_train/results.csv` (final epoch 50):

| Metric | Value |
|--------|-------|
| **Precision** | 89.01% |
| **Recall** | 81.64% |
| **mAP@50** | 87.04% |
| **mAP@50-95** | 51.40% |
| **Training time** | ~20 min (50 epochs) |

### Training Curve (selected epochs)

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1     | 54.83%    | 51.36% | 46.10%| 17.64%   |
| 10    | 82.41%    | 62.63% | 71.44%| 37.53%   |
| 25    | 83.61%    | 73.73% | 81.13%| 45.86%   |
| 40    | 88.59%    | 79.64% | 85.55%| 48.33%   |
| 50    | 89.01%    | 81.64% | 87.04%| 51.40%   |

---

## Inference (Detection on Videos)

1. Open `Ball_Detection.ipynb` in Jupyter.
2. Ensure `ball_best.pt` exists (from training).
3. Place videos in `labelled/` and `labelled 2/` (or adjust paths).
4. Run all cells. Output videos are saved to `detection_output/`.

The notebook:
- Collects videos recursively from the input folders
- Samples N videos per subfolder for testing
- Runs detection with confidence threshold 0.5
- Draws circles and boxes around detected balls
- Previews results with matplotlib

---

## Project Structure

```
.
├── README.md
├── train_ball_model.py      # Training script
├── Ball_Detection.ipynb     # Detection notebook
├── ball_best.pt             # Trained model (best checkpoint)
├── ball_dataset/            # YOLO dataset (generated)
│   ├── data.yaml
│   ├── images/train, images/val
│   └── labels/train, labels/val
├── runs/ball_train/         # Training outputs
│   ├── results.csv          # Metrics per epoch
│   ├── args.yaml            # Training arguments
│   └── weights/best.pt      # Best weights (also copied to ball_best.pt)
├── train/                   # Source COCO dataset
│   ├── _annotations.coco.json
│   └── *.jpg
├── labelled/                # Input videos
├── labelled 2/              # Additional input videos
└── detection_output/        # Output annotated videos
```

---

## Usage Summary

1. **Train**: `python train_ball_model.py` → produces `ball_best.pt`
2. **Detect**: Run `Ball_Detection.ipynb` → produces `*_detected.mp4` in `detection_output/`

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) — Cricket Dataset
- OpenCV, PyTorch
