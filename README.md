# Automated Pneumonia and COVID-19 Detection from Chest X-Ray Images

An end-to-end deep learning system for **3-class chest X-ray classification** (Normal / Pneumonia / COVID-19) using **PyTorch**, with **ensemble predictions**, **MC Dropout uncertainty**, **Grad-CAM explainability**, and an **interactive Streamlit web application**.

---

## Project Highlights

- 3-class classification: **Normal**, **Pneumonia**, **COVID-19**
- Transfer learning with **DenseNet121**, **ResNet50**, and **EfficientNetB0**
- **Ensemble** predictions via AUC-weighted soft voting
- **MC Dropout** uncertainty estimation (aleatoric + epistemic)
- **Grad-CAM** explainability with class activation heatmaps
- **Focal Loss** + inverse-frequency class weights for imbalanced data
- **Confidence threshold** control for sensitivity/specificity trade-off
- **Batch inference** with CSV export
- Interactive **Streamlit web application**

---

## Project Structure

```
.
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── setup_venv.bat                  # Notebook training environment setup (Windows)
├── setup_venv.sh                   # Notebook training environment setup (Linux/Mac)
├── verify_gpu.py                   # CUDA/GPU verification script
├── README.md
├── model/
│   ├── densenet_3class.pt
│   ├── resnet_3class.pt
│   ├── efficientnet_3class.pt
│   └── config.json
├── test_images/                    # One sample X-ray per class for app demo
│   ├── IM-0023-0001.jpeg           #   Normal
│   ├── person1946_bacteria_4875.jpeg #   Pneumonia
│   └── x-ray-image-2b_full.jpg    #   COVID-19
└── notebooks/
    └── Automated_Pneumonia_Detection_from_Chest_X_Ray_Images_pytorch.ipynb
```

---

## Dataset

- **Sources:**
  - Kaggle – [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.)
  - Kaggle – [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Classes:** NORMAL, PNEUMONIA, COVID
- **Split:** Stratified 80 / 10 / 10 (train / val / test)
- **Class imbalance:** handled via `compute_class_weight` + Focal Loss

---

## Models

All three models use torchvision pretrained weights with a custom classification head:

```
BatchNorm1d → Dropout(0.4) → Linear(256) → ReLU → BatchNorm1d → Dropout → Linear(3)
```

| Model | Backbone | Grad-CAM Layer |
|---|---|---|
| DenseNet121 | Dense feature reuse | `denseblock4` |
| ResNet50 | Residual connections | `layer4` |
| EfficientNetB0 | Compound scaling | `features[7]` |

### Training Pipeline

1. Download datasets via Kaggle API
2. Build unified 3-class dataset with stratified split
3. Focal Loss (gamma=2, alpha=0.25) + inverse-frequency class weights
4. `torchvision.ImageFolder` + `DataLoader` with augmentation (`albumentations`)
5. Two-phase transfer learning:
   - Phase 1: frozen backbone — 5 epochs, lr=1e-3
   - Phase 2: unfreeze last block — 20 epochs, lr=1e-5
6. Mixed precision via `torch.cuda.amp.GradScaler`
7. EarlyStopping + ReduceLROnPlateau
8. Ensemble: AUC-weighted soft voting across all three models

---

## Model Performance Summary

| Model | Accuracy | Macro ROC-AUC | Macro F1 | F1-COVID |
|---|---|---|---|---|
| DenseNet121 | 90.19% | 0.9814 | 0.8937 | 0.8678 |
| ResNet50 | 92.63% | 0.9875 | 0.9232 | 0.9131 |
| EfficientNetB0 | 93.67% | 0.9900 | 0.9376 | 0.9407 |
| **Ensemble** | **93.74%** | **0.9908** | **0.9396** | **0.9465** |

> Evaluated on 2,701 held-out test images. High-confidence subset (55.2%, uncertainty < 0.35): **99.40% accuracy**. See the notebook for full confusion matrices, ROC curves, and calibration plots.

---

## Explainability

**Grad-CAM** heatmaps highlight lung regions that drive each model's prediction, registered via PyTorch forward/backward hooks. Available for all three models directly in the Streamlit app.

**MC Dropout** (50 forward passes at inference) provides:
- Per-class probability mean and standard deviation
- Combined uncertainty score — predictions flagged for review if combined uncertainty > 0.35

---

## Streamlit Application

### Features

- Single image inference with per-class probability bar chart
- MC Dropout confidence interval and uncertainty display
- Grad-CAM overlay toggle (per model)
- Side-by-side model comparison
- Confidence threshold slider
- Batch upload with comparison table and CSV export

> See **Setup Guide → Option A** below for full instructions.

---

## Setup Guide

There are **two separate setup paths** depending on what you want to do:

---

### Option A — Run the Streamlit app only (no training)

Use this if you already have the trained `.pt` model files and just want to run the web application.
Uses `requirements.txt`, which installs a lighter set of dependencies (no Kaggle API, no Jupyter).

```bash
# Windows
pip install -r requirements.txt
streamlit run app.py

# Linux / Mac
pip install -r requirements.txt
streamlit run app.py
```

> **Requirements:** Python 3.10+, CUDA-capable GPU recommended (CPU inference also works but is slower).

---

### Option B — Retrain from scratch (notebook)

Use this if you want to reproduce the full training pipeline: dataset download, preprocessing,
model training, evaluation, and export. `setup_venv.bat` / `setup_venv.sh` creates a dedicated
virtual environment with **PyTorch + CUDA 12.4**, registers a Jupyter kernel, and installs all
training dependencies. Requires a CUDA-capable GPU and Kaggle API credentials.

**Windows:**
```bash
# 1. Create venv and install PyTorch cu124 + all training dependencies
setup_venv.bat

# 2. Add Kaggle API credentials
#    Place kaggle.json in C:\Users\<you>\.kaggle\kaggle.json

# 3. Verify GPU
.venv\Scripts\python verify_gpu.py

# 4. Open and run the training notebook
call .venv\Scripts\activate
jupyter notebook notebooks\Automated_Pneumonia_Detection_from_Chest_X_Ray_Images_pytorch.ipynb
```

**Linux / Mac:**
```bash
# 1. Create venv and install PyTorch cu124 + all training dependencies
bash setup_venv.sh

# 2. Add Kaggle API credentials
#    Place kaggle.json in ~/.kaggle/kaggle.json

# 3. Verify GPU
.venv/bin/python verify_gpu.py

# 4. Open and run the training notebook
source .venv/bin/activate
jupyter notebook notebooks/Automated_Pneumonia_Detection_from_Chest_X_Ray_Images_pytorch.ipynb
```

> **Requirements:** Python 3.10+, CUDA-capable GPU (tested on RTX 3060 Laptop, CUDA 12.4 / Driver 13.1), Kaggle API token.

---

## Stack

| Category | Libraries |
|---|---|
| Deep learning | PyTorch 2.3, torchvision |
| Data | NumPy, Pandas, scikit-learn |
| Augmentation | albumentations |
| Image processing | OpenCV (headless), Pillow |
| Visualization | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Dataset | Kaggle API |

---

## Deployment

Compatible with **Streamlit Community Cloud** and **Hugging Face Spaces**.
Ensure trained `.pt` model files and `config.json` are present under `model/` before deploying.

---

## Disclaimer

This project is intended **strictly for educational and research purposes**.
It must **not** be used as a substitute for professional medical diagnosis.

---

## Author

**Rajib Roy** — SR No: 24459 | IISc Bengaluru | DS 216o: Applied AI in Healthcare
