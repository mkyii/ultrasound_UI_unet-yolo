# Real-time ROSC Monitoring Pipeline

Real-time carotid ultrasound analysis framework for ROSC (Return of Spontaneous Circulation) monitoring during CPR.

The pipeline combines:

- YOLO-based vessel detection
- UNet-based vessel segmentation
- Morphological feature extraction
- XGBoost-based ROSC classification
- Real-time visualization dashboard

This framework is designed for continuous monitoring during chest compressions without interrupting CPR.

---

# Pipeline Overview

```text
Input Ultrasound Frame
        ↓
YOLO Detection
(Artery / Vein bbox)
        ↓
UNet Segmentation
(Vessel mask extraction)
        ↓
Morphological Feature Extraction
(CAC / eccentricity / area / roundness ...)
        ↓
XGBoost Classification
(ROSC vs Arrest)
        ↓
Real-time Visualization Dashboard
```

---

# Project Structure

```text
project/
│
├── main.py
├── CFG.py
│
├── models/
│   └── loader.py
│
├── inference/
│   ├── detector.py
│   ├── segmenter.py
│   ├── postprocess.py
│   └── worker.py
│
├── visualization/
│   └── dashboard.py
│
├── utils/
│   ├── io.py
│   ├── metric.py
│   └── features.py
│
├── checkpoint/
│   ├── emr_yolo.pt
│   ├── best_unet.bin
│   └── xgb_models/
│
└── Dataset/
```

---

# Key Features

## 1. YOLO-based Vessel Detection

- Detects carotid artery and internal jugular vein
- Class-wise top-1 confidence filtering
- Bounding-box padding for stable crop extraction

---

## 2. UNet-based Vessel Segmentation

- EfficientNet-B1 encoder
- Crop-based segmentation
- Connected-component filtering
- Hole filling and morphology refinement

---

## 3. Real-time ROSC Classification

Morphological features include:

- vessel area
- major/minor axis
- eccentricity
- axis ratio
- roundness

XGBoost classifier predicts:

- ROSC
- Arrest

---

## 4. Warm-up State Logic

Monitoring starts only after:

```text
IJV eccentricity >= 1.0
```

Before that:

- classification continues internally
- predictions accumulate
- UI status remains in `WARM UP`

This stabilizes the monitoring state during initial probe positioning.

---

## 5. Asynchronous Real-time Pipeline

The system uses:

- UI thread
- background inference worker

to reduce visualization latency during segmentation and classification.

---

# Installation

## Requirements

- Python 3.10+
- CUDA-enabled GPU recommended

---

# Required Packages

```text
torch
torchvision
opencv-python
numpy
pandas
scikit-learn
joblib
ultralytics
segmentation-models-pytorch
```

---

# Run

```bash
python main.py
```

---

# Output Files

## Frame-level CSV

```text
frame_level_eval.csv
```

Contains:

- vessel features
- ROSC probability
- majority probability
- frame prediction
- warm-up state

---

## Patient-level CSV

```text
patient_level_eval.csv
```

Contains:

- patient-level majority voting result
- aggregated ROSC prediction

---

## Visualization Video

```text
time_monitor_all_images.avi
```

Includes:

- YOLO bbox
- vessel segmentation overlay
- ROSC/Arrest status
- CAC graph
- estimated blood pressure trends

---

# Notes

- The system is optimized for real-time CPR ultrasound analysis.
- Segmentation and classification run asynchronously.
- Warm-up frames are hidden from UI status display but still contribute to cumulative prediction.

---

# License

This project is intended for research purposes only.
