# 🫁 Chest X-Ray Pneumonia Detection Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Hybrid-green?style=for-the-badge)

**A hybrid CNN + XGBoost deep learning system for automated pneumonia detection from chest X-ray images.**

*School of Computer Engineering — Manipal Institute of Technology, Manipal*

</div>

---

## 📄 Research Paper

This project is accompanied by a formal research report:  
📎 **[Comparative Analysis of SVM, CNN, and XGBoost for Binary and Multiclass Classification](./Comparative_Analysis_of_SVM_CNN_and_XGBoost_for_Binary_and_Multiclass_Classification.pdf)**

The paper covers the full methodology, dataset analysis, experimental results, and comparative evaluation of SVM, CNN, and hybrid CNN + XGBoost models.

---

## 🧠 Project Overview

Pneumonia is one of the leading causes of mortality worldwide, especially in children and elderly patients. Manual interpretation of chest X-rays is time-consuming and prone to human error — particularly in resource-constrained healthcare settings.

This project builds an **automated, AI-assisted diagnostic system** that classifies chest X-ray images as either **Normal** or **Pneumonia** using a hybrid deep learning + machine learning pipeline.

### Key Highlights

- **Hybrid architecture**: CNN (VGG16 transfer learning) for feature extraction + XGBoost for classification
- **Best validation log loss**: 0.106 (outperforms standalone CNN and SVM)
- **Convergence**: ~72 XGBoost boosting rounds — faster and more stable than CNN alone
- **Streamlit web app** for real-time, demo-friendly inference
- **Explainability**: Lung visualization overlay for demo-grade heatmap output

---

## 🗂️ Repository Structure

```
📦 IAI Project/
├── app.py                        # Streamlit entry point
├── pneumonia_app/
│   ├── inference.py              # Model loading, preprocessing, prediction, stats
│   └── visuals.py                # Lung overlay image and stylized card UI
├── pneumonia_model (2).keras     # Saved Keras model bundle
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📊 Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Class | Images |
|-------|--------|
| Pneumonia | 3,875 |
| Normal | 1,341 |
| **Total** | **5,856** |

- Format: Grayscale JPEG
- Original resolution: ~2000 × 2000 px (resized during preprocessing)
- Class imbalance handled via data augmentation

**Visual patterns:**
- 🔴 Pneumonia → opacities / white patches in lung regions
- 🟢 Normal → clear lung fields with minimal obstruction

---

## ⚙️ Methodology

### Preprocessing Pipeline
- Image resizing and normalization
- Grayscale conversion
- Noise reduction
- Data augmentation (flips, rotations, zoom) to combat overfitting

### Models Evaluated

| Method | Feature Extraction | Interpretability | Compute Cost | Data Requirement |
|--------|--------------------|-----------------|--------------|-----------------|
| SVM | Manual | High | Low–Moderate | Small–Moderate |
| CNN (VGG16) | Automatic | Low | High | Large |
| **CNN + XGBoost (Hybrid)** | **Automatic + ML** | **Moderate–High** | **Moderate–High** | **Moderate–Large** |

### System Pipeline (Hybrid Model)

```
Input X-Ray
    ↓
Load & Preprocess (resize, normalize, augment)
    ↓
Split Dataset (train / val / test)
    ↓
VGG16 CNN — Feature Extraction
    ↓
Flatten Feature Maps → 1D Vectors
    ↓
XGBoost Classifier
    ↓
Predict: Normal / Pneumonia + Evaluate
```

---

## 📈 Results

### SVM
- Best accuracy ~97% at 80/20 split
- Performance degrades significantly with less training data — limited generalization

### CNN (Transfer Learning — VGG16)
- Best split: 60/5/35
- Validation Accuracy: **92.01%**
- Validation Loss: **0.2316**
- ROC-AUC: **~0.90**
- ⚠️ Training instability: erratic validation loss spikes suggesting overfitting / learning rate issues

### Hybrid CNN + XGBoost ✅ Best Model

| Variant | Log Loss |
|---------|----------|
| Raw XGBoost | ~0.146 |
| XGBoost + PCA | ~0.109 |
| **CNN + XGBoost (Hybrid)** | **0.106** |

- Faster convergence: ~72 boosting rounds
- Most stable training behavior
- Recommended for real-world deployment

---

## 🖥️ Streamlit App

A local Streamlit application ships with this repository, making it easy to run inference on any chest X-ray image.

### Features
- Loads saved Keras model (`pneumonia_model (2).keras`) with automatic fallback to `config.json` + `model.weights.h5`
- Reproduces the full notebook preprocessing pipeline
- Outputs: **Normal** vs **Pneumonia** prediction with confidence
- Displays model architecture details and reported performance metrics
- Renders a stylized lung visualization / heatmap overlay for demo explainability

> ⚠️ The heatmap overlay is intended for demo explainability purposes, not medical-grade lesion localization.

### Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

---

## 🔬 Discussion

### Why Hybrid Outperforms Standalone CNN
- **Better generalization**: CNN extracts robust spatial features; XGBoost handles the classification boundary more stably
- **Lower log loss**: Reduced prediction uncertainty — critical for medical applications
- **Less overfitting**: Boosted trees regularize the final classification step
- **Data efficiency**: High accuracy achievable even with moderate dataset sizes, since pneumonia vs. normal class boundaries are well-defined

### Limitations
- Binary classification only (Normal vs. Pneumonia) — no multi-disease support
- Performance tied to dataset quality and size
- May not generalize equally across diverse patient demographics or equipment

---

## 🚀 Future Work

- [ ] Train on larger, more diverse datasets
- [ ] Extend to multi-class lung disease detection
- [ ] Deploy as a production web or mobile application
- [ ] Integrate medical-grade Grad-CAM for lesion localization
- [ ] Hyperparameter tuning for further accuracy gains

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
