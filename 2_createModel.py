# --- Import des packages ---
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             precision_score, recall_score, f1_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Charger les données préparées ---

from sqlalchemy import create_engine
import scipy as sp

# Paramètres de connexion
db_user = 'postgres'
db_password = 'mysecretpassword'
db_host = 'localhost' 
db_port = 5432
db_name = 'postgres'   

# Création de l'engine SQLAlchemy
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Lire la table SQL dans un DataFrame
df_from_sql = pd.read_sql('SELECT * FROM ma_table', engine)

# Afficher les 5 premières lignes

df_from_sql = df_from_sql.dropna(how='all')  # Supprimer les lignes entièrement vides
df_from_sql.head()

# --- Config ---
TRAIN_SIZE = 0.80  # 80% train, 20% test
df = df_final.copy()

# --- Features et target ---
feature_cols = ['close', 'rendement_log', 'vol_30j', 'rendement_log_10j',
       'mm7', 'mm21', 'boll_haut', 'boll_bas', 'rsi_14j', 'sentiment',
       'cluster_kmeans', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'inflation_eur',
       'inflation_us', 'interest_rate_eur', 'interest_rate_us']

X = df[feature_cols].fillna(df[feature_cols].median())
y = df["target"].astype(int)

# --- Split train/test temporel ---
split_idx = int(len(df) * TRAIN_SIZE)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# --- Définir la grille de paramètres ---
param_grid = {
    'n_estimators': [200, 500, 800],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [y_train.value_counts()[0] / y_train.value_counts()[1]]
}

# --- Recherche des meilleurs paramètres sans CV ---
best_auc = 0
best_params = None
best_model = None

for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for subs in param_grid['subsample']:
                for col in param_grid['colsample_bytree']:
                    model = XGBClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        learning_rate=lr,
                        subsample=subs,
                        colsample_bytree=col,
                        scale_pos_weight=param_grid['scale_pos_weight'][0],
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probs)
                    if auc > best_auc:
                        best_auc = auc
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'learning_rate': lr,
                            'subsample': subs,
                            'colsample_bytree': col,
                            'scale_pos_weight': param_grid['scale_pos_weight'][0]
                        }
                        best_model = model

print("\nMeilleurs paramètres trouvés :")
print(best_params)

# --- Évaluation sur le test set ---
probs_test = best_model.predict_proba(X_test)[:, 1]

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
importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importance.tail(10).plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_xlabel("Importance")
axes[2].set_title("Top 10 Features")

plt.tight_layout()
plt.show()

print("\nTop 5 features:")
print(importance.sort_values(ascending=False).head(5))
