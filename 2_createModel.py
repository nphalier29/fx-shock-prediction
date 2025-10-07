# --- Import de packages ---

import pandas as pd
import taceconomics
from datetime import datetime
import numpy as np

from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

"""
XGBoost pour prédiction d'un choc à 2 semaines sur EUR/USD.
"""

import warnings
warnings.filterwarnings("ignore")

# --- config ---

TRAIN_SIZE = 0.80  # 80% train, 20% test
df = df_final.copy()

# --- features et target ---

# Features
feature_cols = ['close', 'rendement_log', 'vol_30j', 'rendement_log_10j',
       'mm7', 'mm21', 'boll_haut', 'boll_bas', 'rsi_14j', 'sentiment',
       'cluster_kmeans', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'inflation_eur',
       'inflation_us', 'interest_rate_eur', 'interest_rate_us']

X = df[feature_cols].fillna(df[feature_cols].median())
y = df["target"].astype(int)

print(f"Features: {X.shape}")

# --- split train/test temporel ---

split_idx = int(len(df) * TRAIN_SIZE)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Target train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# --- entraînement XGBoost ---

print("\nEntraînement XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# --- prédictions et évaluation ---

print("\nÉvaluation sur test...")

# Probabilités
probs_test = model.predict_proba(X_test)[:, 1]

# Métriques ROC
auc = roc_auc_score(y_test, probs_test)
gini = 2 * auc - 1
fpr, tpr, thresholds = roc_curve(y_test, probs_test)

# Seuil optimal (Youden)
youden_idx = np.argmax(tpr - fpr)
threshold = thresholds[youden_idx]

# Prédictions binaires
preds = (probs_test >= threshold).astype(int)

# Métriques
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
accuracy = accuracy_score(y_test, preds)

print(f"\n{'='*50}")
print(f"AUC:        {auc:.4f}")
print(f"Gini:       {gini:.4f}")
print(f"Seuil:      {threshold:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TN={tn:4d}  FP={fp:4d}")
print(f"  FN={fn:4d}  TP={tp:4d}")
print(f"\nPrecision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"{'='*50}")

# --- Visualisations ---

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC Curve
axes[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].scatter(fpr[youden_idx], tpr[youden_idx], c='red', s=100, zorder=5)
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Matrice de confusion
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_ylabel("True")
axes[1].set_xlabel("Predicted")

# Feature Importance
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importance.tail(10).plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_xlabel("Importance")
axes[2].set_title("Top 10 Features")

plt.tight_layout()
plt.show()

print("\nTop 5 features:")
print(importance.sort_values(ascending=False).head(5))

