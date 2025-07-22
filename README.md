# 🧠 Breast Cancer Classification with Machine Learning

This project explores and compares multiple machine learning models for breast cancer diagnosis using the UCI Breast Cancer Wisconsin dataset. It includes full preprocessing, model training, evaluation, explainability with SHAP, and energy consumption tracking using CodeCarbon.

## 🚀 Overview

- 📊 Dataset: UCI Breast Cancer Wisconsin
- 🔍 Models: Logistic Regression, Random Forest, Support Vector Machine
- 📈 Evaluation: Accuracy, classification report, confusion matrix
- 🌱 Sustainability: Energy footprint tracking with CodeCarbon
- 📌 Explainability: SHAP visualizations (beeswarm, bar, waterfall)
- 📂 Output: All visualizations saved in `/plots/` folder

## 🧰 Technologies Used

- **Languages**: Python  
- **Libraries**: scikit-learn, pandas, NumPy, matplotlib, shap, CodeCarbon, ucimlrepo  
- **Tools**: VS Code, GitHub  
- **Concepts**: Classification, Explainable AI, Feature Importance, Environmental Impact

## 📊 Results

Each model was evaluated on a test set (20%) with SHAP values visualized for interpretability. CO₂ emissions during model training were recorded for sustainability tracking.

Example outputs (saved in `/plots/`):
- `shap_beeswarm_random_forest.png`
- `shap_bar_logistic_regression.png`
- `random_forest_feature_importance.png`
- `shap_waterfall_svm.png`

## 📁 Structure

```
breast-cancer-ml/
├── breast_cancer_classification.py
├── README.md
├── requirements.txt
└── plots/
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python breast_cancer_classification.py
```

## 🧪 What I Learned

- How to evaluate and compare ML models on real-world datasets
- Using SHAP for model explainability
- Estimating CO₂ emissions with CodeCarbon
- Good practices for structuring and documenting projects

---

> ⚠ **Note:** The `codecarbon` library may not be compatible with all GPU configurations (e.g., NVIDIA MX series). This project was tested on a desktop with a compatible GPU.

✅ *This project was developed as part of my studies in Electrical and Computer Engineering at DUTH, showcasing applied machine learning for biomedical data analysis.*
