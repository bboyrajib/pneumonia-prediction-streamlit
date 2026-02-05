# 🫁 Automated Pneumonia Detection from Chest X-Ray Images

An end-to-end deep learning system for **automated pneumonia detection** from chest X-ray images using **DenseNet121** and **ResNet50**, with **explainability (Grad-CAM)**, **model comparison**, **batch inference**, and an **interactive Streamlit web application**.

---

## 🚀 Project Highlights

- ✅ Binary classification: **Pneumonia vs Normal**
- 🧠 Transfer learning with **DenseNet121** & **ResNet50**
- 📊 Evaluation metrics: Accuracy, ROC-AUC, Sensitivity, Specificity
- 🔍 **Grad-CAM explainability** for clinical interpretability
- 🔄 **Side-by-side model comparison**
- 📂 **Batch inference** with CSV export
- 🎛️ **Confidence threshold control**
- 🌐 **Streamlit web application**
- ☁️ Ready for **Streamlit Cloud deployment**

---

## 📁 Project Structure

```
.
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── model/
│   ├── densenet_pneumonia_model.keras
│   └── resnet_pneumonia_model.keras
├── notebooks/
│   └── pneumonia_detection.ipynb   # Training & evaluation notebook
└── .streamlit/
    └── config.toml                 # Optional theme configuration
```

---

## 🗂 Dataset

- **Source:** Kaggle – Chest X-Ray Pneumonia Dataset
- **Images:** Pediatric chest X-rays
- **Classes:** Normal, Pneumonia

### Dataset Split

| Split | Normal | Pneumonia | Total |
|-----|--------|-----------|-------|
| Train | 1341 | 3875 | 5216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

---

## 🧠 Models Used

### DenseNet121
- Dense feature reuse via dense connections
- Higher ROC-AUC and sensitivity
- Efficient and well-calibrated probabilities

### ResNet50
- Residual learning for deep gradient flow
- Strong localized Grad-CAM activations
- Useful comparison baseline

---

## 📈 Model Performance Summary

| Model | Accuracy | ROC-AUC | Sensitivity | Specificity |
|-----|---------|--------|------------|------------|
| DenseNet121 | 0.88 | 0.95 | 0.91 | 0.85 |
| ResNet50 | 0.78 | 0.84 | 0.84 | 0.68 |

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM heatmaps highlight regions in the lungs that contribute most to model predictions.

- Improves **model transparency**
- Helps validate **clinical relevance**
- Available for both DenseNet121 and ResNet50
- Downloadable from the Streamlit app

---

## 🎛️ Confidence Threshold

The confidence threshold controls how model probabilities are converted into class labels.

- Lower threshold → higher sensitivity (screening use-case)
- Higher threshold → higher specificity (confirmatory analysis)

This allows safe exploration of sensitivity–specificity trade-offs.

---

## 📂 Batch Inference

- Upload multiple X-ray images simultaneously
- Predictions from **both models**
- Agreement / disagreement analysis
- Confidence difference (Δ)
- Export results as CSV

Designed to simulate **real-world screening workflows**.

---

## 🌐 Streamlit Web Application

### Features
- Single image prediction
- Side-by-side model comparison
- Grad-CAM toggle
- Confidence threshold slider
- Batch upload & CSV export
- Card-based, user-friendly UI

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

The application is compatible with:
- **Streamlit Community Cloud**
- **Hugging Face Spaces**

Ensure model files are present under the `model/` directory before deployment.

---

## ⚠️ Disclaimer

> This project is intended **strictly for educational and research purposes**.
> It must **not** be used as a substitute for professional medical diagnosis.

---

## 🔮 Future Enhancements

- Ensemble predictions
- Automatic threshold optimization
- Probability calibration visualization
- Multi-class lung disease detection
- Integration with clinical metadata

---

## 👨‍💻 Author

Developed as a full-stack **machine learning + explainability + deployment** project, focusing on real-world usability and interpretability in medical imaging.
