# PPE Detection Model Setup Guide

## 1. Why a Specialized PPE Model is Mandatory

Standard pretrained object detection models (such as default YOLOv8 trained on the 80-class Microsoft COCO dataset) **cannot detect safety helmets or high-visibility vests**. The COCO dataset only contains classes such as `person`, `car`, `dog`, etc.

To perform real PPE compliance detection, this system requires a YOLO model fine-tuned on workplace safety datasets containing:
- `person` (or `worker`)
- `helmet` (or `hard_hat`)
- `no_helmet` (or `head`)
- `vest` (or `safety_vest`)
- `no_vest` (or `no_safety_vest`)

---

## 2. Option A: Download a Free Pretrained PPE Model

You can obtain ready-to-use YOLOv8 weights from Roboflow Universe or GitHub open-source repositories:

1. Visit [Roboflow Universe: Construction Site Safety / PPE Datasets](https://universe.roboflow.com/search?q=ppe+detection)
2. Search for: **"Worker Safety PPE"** or **"Hardhat and Safety Vest Detection"**
3. Select an open dataset and choose **Export > YOLOv8 PyTorch**
4. If weights are provided, download the `best.pt` file.
5. Copy the downloaded file into this folder:
   ```
   smart-ppe-detection/models/best.pt
   ```

---

## 3. Option B: Train Your Own YOLOv8 Model (5-10 Minutes with GPU/Colab)

If you have a Google Colab notebook or a local NVIDIA GPU:

```python
# Install Ultralytics
pip install ultralytics

# Train YOLOv8 nano on your PPE dataset
from ultralytics import YOLO

# Load lightweight pretrained backbone
model = YOLO('yolov8n.pt')

# Train on your custom dataset
model.train(
    data='ppe_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='ppe_yolov8_model'
)
```

Once training completes, locate the trained weights at `runs/detect/ppe_yolov8_model/weights/best.pt` and move it to `smart-ppe-detection/models/best.pt`.

---

## 4. How to Verify Your Model Classes

Run this one-line Python check in your terminal to inspect the class names inside your `best.pt` file:

```bash
python -c "from ultralytics import YOLO; m = YOLO('models/best.pt'); print('Model Classes:', m.names)"
```

The system's `src/compliance.py` module automatically normalizes aliases such as:
- `hard_hat`, `hardhat`, `helmet`, `safety_helmet` &rarr; `helmet`
- `no_helmet`, `no-helmet`, `head` &rarr; `no_helmet`
- `vest`, `safety_vest`, `protective_suit`, `jacket` &rarr; `vest`
- `no_vest`, `no-vest`, `without_vest` &rarr; `no_vest`

---

## 5. Built-in Diagnostic Fallback Mode

If you do not have `best.pt` yet while testing your software or presenting your project:
The application includes a built-in **Diagnostic Fallback Mode** that detects face and upper-body regions and performs color-histogram checks (neon yellow/orange for vests, white/yellow for helmets) or simulated bounding boxes so the entire web interface, SQLite database, audio alerts, and ESP32 hardware signals work end-to-end without crashing!
