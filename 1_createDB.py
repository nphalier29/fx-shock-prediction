import pandas as pd
import taceconomics
from datetime import datetime

# APIKEY
taceconomics.api_key = "sk_o24BhJRqVpIvxVSXX3yiKRGgpDEejmyJ8pfLFX2q22s"  
start_date = '2023-01-01'
end_date = datetime.today().strftime("%Y-%m-%d")

# --- Importation des données ---

# Taux de change EUR/USD
usd_eur = taceconomics.getdata(f"EXR/EUR/WLD?start_date={start_date}")
usd_eur.columns = ["usd_eur"]
usd_eur = usd_eur.dropna()
eur_usd = 1 / usd_eur  # Inversion pour avoir EUR/USD
eur_usd.columns = ["eur_usd"]
eur_usd.index = pd.to_datetime(eur_usd.index)

# --- Indicateurs quantitatifs ---

# Taux de croissance
eur_usd["taux_croissance"] = eur_usd["eur_usd"].pct_change() * 100

# Vol
eur_usd["vol"] = eur_usd["taux_croissance"].rolling(window=30).std()

# rendement à 10j (2 semaines en jours ouvrés)
eur_usd["rendement_10j"] = eur_usd["eur_usd"].pct_change(periods=10) * 100

# --- Cible --- 

# Seuil de choc (±2σ)
eur_usd["seuil_haut"] = 2 * eur_usd["vol"]
eur_usd["seuil_bas"] = -2 * eur_usd["vol"]

# Variable cible : 1 si choc (hausse ou baisse), 0 sinon
eur_usd["target"] = ((eur_usd["taux_croissance"] >= eur_usd["seuil_haut"]) | (eur_usd["taux_croissance"] <= eur_usd["seuil_bas"])).astype(int)

eur_usd[["taux_croissance", "vol", "target"]].tail(10)


# --- Indicateurs chartistes pour visualisation ---

# Moyennes mobiles
eur_usd["moyenne_mobile_7j"] = eur_usd["eur_usd"].rolling(window=7).mean()
eur_usd["moyenne_mobile_21j"] = eur_usd["eur_usd"].rolling(window=21).mean()

# Bandes de Bollinger
eur_usd["bollinger_moyenne"] = eur_usd["eur_usd"].rolling(window=20).mean()
eur_usd["bollinger_haut"] = eur_usd["bollinger_moyenne"] + 2 * eur_usd["eur_usd"].rolling(window=20).std()
eur_usd["bollinger_bas"] = eur_usd["bollinger_moyenne"] - 2 * eur_usd["eur_usd"].rolling(window=20).std()

print(eur_usd.tail())


# RSI (Relative Strength Index)
def calculer_rsi(series, window=14):
    delta = series.diff()
    gains = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    pertes = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gains / pertes
    rsi = 100 - (100 / (1 + rs))
    return rsi

eur_usd["rsi_14j"] = calculer_rsi(eur_usd["eur_usd"])

print(eur_usd.tail())


# --- Indicateur de sentiment ---
from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Télécharger VADER
nltk.download('vader_lexicon')

def recuperer_tous_articles_gdelt(start_date, end_date, keyword="EUR/USD", language='eng', chunk_days=30):
    """
    Récupère tous les articles GDELT entre start_date et end_date
    en faisant des requêtes par tranches de `chunk_days` jours.
    """
    # Initialiser VADER
    sid = SentimentIntensityAnalyzer()

    # Convertir les dates en objets datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialiser GdeltDoc
    gd = GdeltDoc()
    all_articles = []

    # Parcourir par tranches de `chunk_days` jours
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)

        print(f"Récupération des articles entre {current_start.date()} et {current_end.date()}")

        # Créer les filtres pour la tranche actuelle
        f = Filters(
            start_date=current_start.strftime("%Y-%m-%d"),
            end_date=current_end.strftime("%Y-%m-%d"),
            num_records=250,
            keyword=keyword,
            language=language
        )

        # Récupérer les articles
        articles_df = gd.article_search(f)

        # Vérifier si le DataFrame n'est pas vide
        if not articles_df.empty:
            all_articles.append(articles_df)

        # Passer à la tranche suivante
        current_start = current_end + timedelta(days=1)

    # Concaténer tous les DataFrames
    if not all_articles:
        return pd.DataFrame()

    df = pd.concat(all_articles, ignore_index=True)

    # Nettoyer les colonnes utiles et convertir la date
    def convertir_date_gdelt(date_str):
        # Le format est AAAAMMJJTHHMMSSZ, on extrait AAAAMMJJ
        return datetime.strptime(date_str.split('T')[0], "%Y%m%d").date()

    df['date'] = df['seendate'].apply(convertir_date_gdelt)
    df['date'] = pd.to_datetime(df['date'])

    # Calculer le sentiment pour chaque article
    def calculer_sentiment(texte):
        if isinstance(texte, str):
            return sid.polarity_scores(texte)['compound']
        return 0

    df['sentiment'] = df['title'].apply(calculer_sentiment)

    return df


# --- Variables PCA ---

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = eur_usd.copy().dropna().reset_index()

X = df.select_dtypes(include=['float64','int64'])  # uniquement variables numériques
X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=0.9)  # garder assez de composantes pour expliquer 90% de la variance
X_pca = pca.fit_transform(X_scaled)

# Créer un DataFrame avec les nouvelles variables PCA
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

df_extended = pd.concat([df, df_pca], axis=1)


kmeans = KMeans(n_clusters=3, random_state=42)  # exemple avec 3 clusters
df_extended['cluster_kmeans'] = kmeans.fit_predict(X_pca)


print(df_extended.tail(60))

#----Variables macro----

# Inflation

infl_eur = taceconomics.getdata(f"EUROSTAT/EI_CPHI_M_CP-HI00_NSA_HICP2015/EUZ?collapse=M&transform=growth_yoy&start_date={start_date}")
infl_us = taceconomics.getdata(f"FRED/CPIAUCSL/USA?collapse=M&transform=growth_yoy&start_date={start_date}")

df_extended['infl_eur'] = df['infl_eur'].ffill()
df_extended['infl_us'] = df['infl_us'].ffill()
# infl_eur = taceconomics.getdata(f"IFS/PCPIHA_IX_M/EUZ?start_date={start_date}")
# infl_us = taceconomics.getdata(f"IFS/PCPI_IX_M/USA?start_date={start_date}")

df_final = df_extended.join(infl_eur).join(infl_us)

 # Taux interets

ti_eur = taceconomics.getdata(f"ECB/FM_D_EUR_4F_KR_DFR_LEV/EUZ?collapse=M&collapse_mode=end_of_period&start_date={start_date}")
ti_us = taceconomics.getdata(f"DS/USPRATE./WLD?collapse=M&start_date={start_date}")
df_final['ti_eur'] = df['ti_eur'].ffill()
df_final['ti_us'] = df['ti_us'].ffill()


# ti_eur = taceconomics.getdata(f"IFS/FPOLM_PA_M/EUZ?start_date={start_date}")
# ti_us = taceconomics.getdata(f"IFS/FPOLM_PA_M/USA?start_date={start_date}")

df_final = df_final.join(ti_eur).join(ti_us)



print(df_final.tail(40))


# MODELISATION

# xgb_single_model_timeseries.py
"""
XGBoost focalisé pour prédiction d'un choc à 2 semaines sur EUR/USD.
Sorties :
- AUC, Gini, courbe ROC (test),
- seuil optimal (Youden) et matrice de confusion + métriques associées,
- modèle final enregistré (joblib).


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# -----------------------
# Config 
# -----------------------
RANDOM_STATE = 42
# Fractions pour split contigu (doivent sommer à 1.0)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# TimeSeries CV splits (pour GridSearch)
TS_SPLITS = 5

# Grid search params 
XGB_PARAM_GRID = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Metric d'optimisation pour la recherche d'hyperparamètres (ici F1, tu peux changer en 'roc_auc')
GRID_SCORING = 'f1'

# Early stopping rounds pour re-entrainement final
EARLY_STOPPING_ROUNDS = 50

# Fichiers de sortie
MODEL_OUTPATH = "xgb_final_model.joblib"
IMPUTER_OUTPATH = "imputer.joblib"

# -----------------------
# Chargement & checks
# -----------------------
print("Chargement des données...")

df = df_final.sort_values(DATE_COL).reset_index(drop=True)

if DATE_COL not in df.columns:
    raise ValueError(f"Colonne date '{DATE_COL}' introuvable.")
if TARGET_COL not in df.columns:
    raise ValueError(f"Colonne cible '{TARGET_COL}' introuvable.")

# index temporel
df.set_index(DATE_COL, inplace=True)

# Features & target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# On suppose tout numérique — sinon adapter types/catégoriques
num_cols = X.columns.tolist()

# -----------------------
# Split contigu (train / val / test)
# -----------------------
n = len(df)
if not abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8:
    raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC doit être égal à 1.0")

train_end = int(n * TRAIN_FRAC)
val_end = train_end + int(n * VAL_FRAC)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

print(f"Tailles -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

# -----------------------
# Imputation (median) - fit uniquement sur train
# -----------------------
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train[num_cols])

X_train_imp = pd.DataFrame(imputer.transform(X_train[num_cols]), index=X_train.index, columns=num_cols)
X_val_imp   = pd.DataFrame(imputer.transform(X_val[num_cols]), index=X_val.index, columns=num_cols)
X_test_imp  = pd.DataFrame(imputer.transform(X_test[num_cols]), index=X_test.index, columns=num_cols)

# Save imputer for reproducibility / production
joblib.dump(imputer, IMPUTER_OUTPATH)
print(f"Imputer sauvegardé -> {IMPUTER_OUTPATH}")

# -----------------------
# GridSearchCV (TimeSeriesSplit) pour chercher les meilleurs hyperparams
# -----------------------
print("\nLancement GridSearchCV (TimeSeriesSplit) sur XGBoost...")

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)

tscv = TimeSeriesSplit(n_splits=TS_SPLITS)

# GridSearchCV sur les features imputées (pas de scaling nécessaire pour XGBoost)
gscv = GridSearchCV(
    estimator=xgb,
    param_grid=XGB_PARAM_GRID,
    scoring=GRID_SCORING,
    cv=tscv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

gscv.fit(X_train_imp, y_train)
print("Meilleurs paramètres trouvés (GridSearchCV):")
print(gscv.best_params_)
print(f"Best CV {GRID_SCORING}: {gscv.best_score_:.4f}")

# -----------------------
# Ré-entraînement final avec early stopping sur l'échantillon de validation
# -----------------------
best_params = gscv.best_params_.copy()

# Conserver paramètres choisis et activer early stopping via eval_set
xgb_final = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE,
                          n_jobs=-1, **best_params)

print("\nRé-entrainement final avec early stopping sur validation (eval_set)...")
xgb_final.fit(
    X_train_imp, y_train,
    eval_set=[(X_val_imp, y_val)],
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=False
)

# Sauvegarde modèle
joblib.dump(xgb_final, MODEL_OUTPATH)
print(f"Modèle final sauvegardé -> {MODEL_OUTPATH}")

# -----------------------
# Prédiction out-of-sample (test) + évaluation
# -----------------------
print("\nÉvaluation out-of-sample (test)...")
probs_test = xgb_final.predict_proba(X_test_imp)[:, 1]
auc = roc_auc_score(y_test, probs_test)
gini = 2*auc - 1
fpr, tpr, thresholds = roc_curve(y_test, probs_test)

# Seuil optimal - Youden (TPR - FPR maximisé)
youden_idx = np.argmax(tpr - fpr)
opt_threshold_youden = thresholds[youden_idx]

# Seuil optimisant F1 (pour info)
f1_scores = [f1_score(y_test, (probs_test >= t).astype(int), zero_division=0) for t in thresholds]
opt_threshold_f1 = thresholds[np.argmax(f1_scores)]

# Choix du seuil final : tu peux choisir Youden ou F1 ; ici on utilise Youden tout en reportant F1-opt
threshold = opt_threshold_youden

preds = (probs_test >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)            # sensitivity
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
bal_acc = balanced_accuracy_score(y_test, preds)

# -----------------------
# Affichage résultats (out-of-sample uniquement)
# -----------------------
print(f"\nRésultats (test) - XGBoost")
print(f"AUC (ROC): {auc:.4f}")
print(f"Gini: {gini:.4f}")
print(f"Seuil Youden: {opt_threshold_youden:.4f} | Seuil F1-opt: {opt_threshold_f1:.4f}")
print("Matrice de confusion (tn, fp, fn, tp):", (int(tn), int(fp), int(fn), int(tp)))
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")

# -----------------------
# Tracer ROC (test)
# -----------------------
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")
plt.plot([0,1],[0,1], linestyle='--', alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - Out-of-sample (test)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# Tracer Matrice de confusion (test)
# -----------------------
cm = np.array([[tn, fp],[fn, tp]])
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Matrice de confusion (test) - seuil={threshold:.4f}")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.show()

# -----------------------
# Importance des features (optionnel, utile pour interprétation)
# -----------------------
try:
    imp = pd.Series(xgb_final.feature_importances_, index=num_cols).sort_values(ascending=False)
    print("\nTop 10 features par importance (XGBoost):")
    print(imp.head(10))
    plt.figure(figsize=(6,4))
    imp.head(15).plot(kind='bar')
    plt.title("Feature importances (XGBoost)")
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# -----------------------
# Résumé final en dictionary (pratique pour reporting programmatique)
# -----------------------
report = {
    'auc': float(auc),
    'gini': float(gini),
    'threshold_youden': float(opt_threshold_youden),
    'threshold_f1': float(opt_threshold_f1),
    'threshold_used': float(threshold),
    'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'precision': float(precision),
    'recall': float(recall),
    'specificity': float(specificity),
    'f1': float(f1),
    'accuracy': float(accuracy),
    'balanced_accuracy': float(bal_acc),
    'best_params': best_params,
    'trained_at': datetime.utcnow().isoformat() + 'Z'
}

# Enregistrer le reporting si souhaité
pd.Series(report).to_json("xgb_report_test.json")
print("\nReport JSON sauvegardé -> xgb_report_test.json")
print("\nTerminé. Seuls les résultats out-of-sample (test) ont été affichés.")
"""

