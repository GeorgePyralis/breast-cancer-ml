# =====================================
# Breast Cancer Classification using ML Models
# =====================================

# === 1. Libraries ===
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import os

# === 2. Load Dataset ===
dataset = fetch_ucirepo(id=17)
X = dataset.data.features
y = dataset.data.targets

# Merge into one DataFrame
df = pd.concat([X, y], axis=1)

# === 3. Preprocessing ===
df.drop(columns=['ID'], errors='ignore', inplace=True)
df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# === 4. Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === 6. Model Definitions ===
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# === 7. Training & Evaluation ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 8. Energy Tracking & SHAP Analysis ===
os.makedirs('plots', exist_ok=True)

for name, model in models.items():
    print(f"\n=== Energy & SHAP Analysis for {name} ===")

    tracker = EmissionsTracker(project_name=f"{name}_energy")
    tracker.start()
    model.fit(X_train, y_train)  # Re-train for energy measurement
    emissions = tracker.stop()
    print(f"CO₂ emissions for {name}: {emissions:.8f} kgCO₂eq")

    # === SHAP Analysis ===
    try:
        print(f"Calculating SHAP for {name}...")
        masker = shap.maskers.Independent(X_test, max_samples=100)
        explainer = shap.PermutationExplainer(model.predict_proba, masker)
        shap_values = explainer(X_test[:20])
        shap_values_class1 = shap_values[:, :, 1]

        # Beeswarm
        shap.plots.beeswarm(shap_values_class1, show=False)
        plt.title(f"SHAP Beeswarm Plot – {name}")
        plt.tight_layout()
        plt.savefig(f"plots/shap_beeswarm_{name.lower().replace(' ', '_')}.png")
        plt.close()

        # Bar
        shap.plots.bar(shap_values_class1.mean(0), show=False)
        plt.title(f"SHAP Bar Plot – {name}")
        plt.tight_layout()
        plt.savefig(f"plots/shap_bar_{name.lower().replace(' ', '_')}.png")
        plt.close()

        # Waterfall
        shap.plots.waterfall(shap_values_class1[0], show=False)
        plt.title(f"SHAP Waterfall Plot – {name}")
        plt.tight_layout()
        plt.savefig(f"plots/shap_waterfall_{name.lower().replace(' ', '_')}.png")
        plt.close()

        print(f"SHAP plots for {name} saved.")

    except Exception as e:
        print(f"SHAP analysis for {name} failed: {e}")

    # === Feature Importance for Random Forest ===
    if name == "Random Forest":
        plt.figure(figsize=(12, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title(f'{name} – Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'plots/{name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()
        print(f"{name} feature importance plot saved.")

print("\nAnalysis for all models completed.")
