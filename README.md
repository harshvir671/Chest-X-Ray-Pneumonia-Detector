# 🫁 Chest X-Ray Pneumonia Detection Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![MobileNet](https://img.shields.io/badge/MobileNet-Transfer%20Learning-blueviolet?style=for-the-badge)
![MIT Manipal](https://img.shields.io/badge/MIT-Manipal-purple?style=for-the-badge)

**A MobileNet-based deep learning system for automated pneumonia detection from chest X-ray images, with a comparative analysis of SVM, CNN, and XGBoost approaches.**

*School of Computer Engineering — Manipal Institute of Technology, Manipal*

</div>

---

## 📄 Research Paper

This project is accompanied by a formal comparative research report:  
📎 **[Comparative Analysis of SVM, CNN, and XGBoost for Binary and Multiclass Classification](./Comparative_Analysis_of_SVM_CNN_and_XGBoost_for_Binary_and_Multiclass_Classification.pdf)**

The paper benchmarks traditional ML (SVM, XGBoost) against deep learning (CNN) and hybrid approaches, providing the theoretical and experimental basis for the final model selection.

---

## 🧠 Project Overview

Pneumonia is one of the leading causes of mortality worldwide, especially in children and elderly patients. Manual interpretation of chest X-rays is time-consuming and prone to human error — particularly in resource-constrained healthcare settings.

This project builds an **automated, AI-assisted diagnostic system** that classifies chest X-ray images as either **Normal** or **Pneumonia**. The deployed model uses **MobileNet** trained from scratch on the Kaggle chest X-ray dataset. A comparative study (documented in the research paper) evaluates SVM, standalone CNN (VGG16), and hybrid CNN + XGBoost approaches to justify the final architecture.

---

## 🗂️ Repository Structure

```
📦 IAI Project/
├── app.py                         # Streamlit entry point
├── chest-x-ray.ipynb              # Training notebook (MobileNet)
├── pneumonia_app/
│   ├── __init__.py
│   ├── inference.py               # Model loading, preprocessing, prediction
│   └── visuals.py                 # Lung overlay and stylized card UI
├── pneumonia_model (2).keras/
│   ├── config.json
│   ├── metadata.json
│   └── model.weights.h5           # Trained MobileNet weights
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Pneumonia | Normal | Total |
|-------|-----------|--------|-------|
| Train (80%) | 3,418 | 1,224 | 4,642 |
| Test (15%) | 641 | 278 | 919 |
| Val (5%) | 214 | 81 | 295 |
| **Total** | **4,273** | **1,583** | **5,856** |

- Format: Grayscale JPEG, resized to 224 × 224 px
- Class imbalance handled via **sklearn `compute_class_weight`**
- Random seed `10` for reproducibility

**Visual patterns:**
- 🔴 Pneumonia → opacities / white patches in lung regions
- 🟢 Normal → clear lung fields with minimal obstruction

---

## ⚙️ Methodology

### Preprocessing Pipeline
- Grayscale → RGB conversion (stacked channels for MobileNet compatibility)
- Resize to 224 × 224 px using cubic interpolation
- Normalize to [0, 1]
- Data augmentation: rotation (7°), width/height shift (5%), shear (0.2), zoom (0.45), horizontal flip

### Model Architecture — MobileNet

```
Input (224 × 224 × 3)
    ↓
MobileNet base (weights=None, trained from scratch)
    ↓
GlobalAveragePooling2D
    ↓
Dense(1, activation='sigmoid')
    ↓
Predict: Normal (0) / Pneumonia (1)
```

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Binary Accuracy, MAE
- Epochs: 64 | Batch size: 32
- Hardware: 2× Tesla T4 GPUs

### Comparative Analysis (Research Paper)

The accompanying paper benchmarks three approaches on the same dataset:

| Method | Feature Extraction | Interpretability | Compute Cost |
|--------|--------------------|-----------------|--------------|
| SVM | Manual | High | Low–Moderate |
| CNN (VGG16) | Automatic | Low | High |
| **CNN + XGBoost (Hybrid)** | **Automatic + ML** | **Moderate** | **Moderate–High** |

---

## 📈 Results

### MobileNet — Deployed Model

| Metric | Score |
|--------|-------|
| Train Accuracy | **96.88%** |
| Test Accuracy | **68.23%** |
| Precision | **99.72%** |
| Recall (Sensitivity) | **54.60%** |
| Specificity | **99.64%** |
| F1-Score | **70.56** |
| AUC-ROC | **0.77** |

**Confusion Matrix:**
```
              Predicted Normal  Predicted Pneumonia
Actual Normal       277                  1
Actual Pneumonia    291                350
```

> 📌 The model achieves near-perfect precision and specificity — meaning it almost never incorrectly flags a healthy patient. The lower recall indicates some pneumonia cases are missed, a known trade-off in high-precision medical classifiers.

### Comparative Results (from Research Paper)

| Model | Key Metric | Notes |
|-------|-----------|-------|
| SVM | ~97% accuracy (80/20 split) | Degrades with less data |
| CNN (VGG16) | 92.01% val accuracy, ROC-AUC ~0.90 | Training instability observed |
| CNN + XGBoost | Log loss: **0.106** | Most stable, best generalization |

---

## 🖥️ Streamlit App

A local Streamlit application ships with this repository for real-time inference.

### Features
- Loads the trained MobileNet model (`pneumonia_model (2).keras`) with automatic fallback to `config.json` + `model.weights.h5`
- Reproduces the full notebook preprocessing pipeline
- Outputs: **Normal** vs **Pneumonia** prediction with confidence score
- Displays model architecture and reported performance metrics
- Renders a stylized lung visualization / heatmap overlay

> ⚠️ The heatmap overlay is for demo explainability only, not medical-grade lesion localization.

### Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/harshvir671/Chest-X-Ray-Pneumonia-Detector.git
cd Chest-X-Ray-Pneumonia-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔬 Discussion

### Training Behavior
The validation loss curve shows characteristic instability — smooth training loss alongside erratic validation spikes — consistent with the training instability noted in the research paper for standalone CNN models. This is largely attributable to the small validation set (295 images) and high learning rate sensitivity in MobileNet without pretrained weights.

### Why High Precision but Lower Recall?
The model is highly conservative: it almost never falsely flags a healthy patient (only 1 false positive out of 278 normal cases). However, it misses 291 out of 641 pneumonia cases. For a screening tool, this trade-off would need tuning via threshold adjustment.

### Limitations
- Binary classification only — no multi-disease support
- Validation set is very small (295 images), causing unstable val metrics
- MobileNet trained from scratch rather than with ImageNet weights
- May not generalize to X-rays from different equipment or demographics

---

## 🚀 Future Work

- [ ] Use ImageNet pretrained MobileNet weights for better initialization
- [ ] Add learning rate scheduling (e.g. ReduceLROnPlateau)
- [ ] Increase validation set size for more stable evaluation
- [ ] Extend to multi-class lung disease detection
- [ ] Integrate Grad-CAM for medical-grade localization
- [ ] Deploy as a production web or mobile application

---

## 🔒 Ethical Considerations

This project uses a publicly available, anonymized dataset. The following principles guide responsible use:

- **Informed Consent**: Patient data should be collected with proper consent
- **Anonymization**: All personal identifiers must be removed
- **Regulatory Compliance**: HIPAA and GDPR standards apply
- **Data Security**: Prevent unauthorized access
- **Transparency**: Data usage limited to research purposes
- **Accountability**: Ethical oversight maintained throughout

> This system is a **decision support tool** and is not intended to replace qualified medical professionals.

---

## 📚 References

1. Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv:1711.05225, 2017.
2. Wang et al., *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks*, CVPR 2017.
3. Szepesi & Szilágyi, *Detection of pneumonia using CNNs and deep learning*, Biocybernetics and Biomedical Engineering, 2022.
4. Chen & Guestrin, *XGBoost: A scalable tree boosting system*, KDD 2016.
5. Hedhoud et al., *An improvement of the CNN-XGBoost model for pneumonia disease classification*, Polish Journal of Radiology, 2023.

---

<div align="center">

*School of Computer Engineering · Manipal Institute of Technology, Manipal*

</div>
