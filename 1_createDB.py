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

# --- API Key et dates ---

taceconomics.api_key = ""  # Insérer votre clé API ici
start_date = '2020-01-01'
end_date = datetime.today().strftime("%Y-%m-%d")

# --- Importation des données ---

# Taux de change EUR/USD
usd_eur = taceconomics.getdata(f"EXR/EUR/WLD?start_date={start_date}")
usd_eur.columns = ["usd_eur"]
usd_eur = usd_eur.dropna()
eur_usd = 1 / usd_eur  # Inversion pour avoir EUR/USD
eur_usd.columns = ["close"]
eur_usd.index = pd.to_datetime(eur_usd.index)

# print(eur_usd.head(10))
# print(eur_usd.shape)
# print(eur_usd.columns)

# --- Indicateurs de base ---

# Rendement logarithmique quotidien
eur_usd["rendement_log"] = np.log(eur_usd["close"] / eur_usd["close"].shift(1))

# Volatilité glissante (30 jours) basée sur les rendements log
eur_usd["vol_30j"] = eur_usd["rendement_log"].rolling(window=30).std()

# Rendement logarithmique cumulé à 10 jours (passé)
eur_usd["rendement_log_10j"] = np.log(eur_usd["close"] / eur_usd["close"].shift(10))

# # print(eur_usd.head(60))
# print(eur_usd.shape)
# print(eur_usd.columns)

# --- Cible --- 

# Rendement logarithmique FUTUR à 10 jours (cohérent avec les autres features)
eur_usd["rendement_futur_10j"] = np.log(eur_usd["close"].shift(-10) / eur_usd["close"])

# Seuils de choc dynamiques (±1 × volatilité sur 10 jours)
# La vol sur 10j est approximativement vol_quotidienne × sqrt(10)
seuil_choc = eur_usd["vol_30j"] * np.sqrt(10) # 1*vol car 2*vol est trop rare (4-5% des cas)

# Target : 1 si choc de volatilité (hausse OU baisse), 0 sinon
eur_usd["target"] = (
    (eur_usd["rendement_futur_10j"].abs() >= seuil_choc)
).astype(int)

# print(eur_usd.shape)
# print(eur_usd.columns)

# Distribution
# print(f"\nDistribution de la target:")
# print(eur_usd['target'].value_counts())
# print(f"Taux de chocs: {eur_usd['target'].mean():.2%}")

# --- Indicateurs chartistes ---

# Moyennes mobiles
eur_usd["mm7"] = eur_usd["close"].rolling(window=7).mean()
eur_usd["mm21"] = eur_usd["close"].rolling(window=21).mean()

# Bandes de Bollinger (20 jours)
rolling_mean_20 = eur_usd["close"].rolling(window=20).mean()
rolling_std_20 = eur_usd["close"].rolling(window=20).std()
eur_usd["boll_haut"] = rolling_mean_20 + 2 * rolling_std_20
eur_usd["boll_bas"] = rolling_mean_20 - 2 * rolling_std_20

# RSI
def calculer_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    perte = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=window).mean()
    avg_perte = pd.Series(perte, index=series.index).rolling(window=window).mean()
    rs = avg_gain / avg_perte
    rsi = 100 - (100 / (1 + rs))
    return rsi

eur_usd["rsi_14j"] = calculer_rsi(eur_usd["close"])

eur_usd = eur_usd.drop(columns=["rendement_futur_10j"])
# eur_usd = eur_usd.dropna().copy()

# # print(eur_usd.head(60))
# print(eur_usd.shape)
# print(eur_usd.columns)
# print(eur_usd.iloc[0])

# --- Indicateurs de sentiment ---

# Télécharger VADER (une seule fois)
nltk.download('vader_lexicon')

# Initialiser VADER globalement
sid = SentimentIntensityAnalyzer()

def recuperer_sentiment_gdelt(start_date, end_date, keyword="EUR/USD", language='eng', chunk_days=30, num_records=250):
    """
    Récupère les articles GDELT pour un mot-clé donné entre deux dates,
    calcule le score de sentiment pour chaque article, et agrège par jour.
    """
    # Convertir les dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    gd = GdeltDoc()
    all_articles = []

    current_start = start
    while current_start <= end:
        current_end = min(current_start + timedelta(days=chunk_days), end)

        # # print(f"Récupération articles: {current_start.date()} -> {current_end.date()}")

        # Créer filtre
        f = Filters(
            start_date=current_start.strftime("%Y-%m-%d"),
            end_date=current_end.strftime("%Y-%m-%d"),
            num_records=num_records,
            keyword=keyword,
            language=language
        )

        # Récupérer articles
        try:
            articles_df = gd.article_search(f)
            if not articles_df.empty:
                all_articles.append(articles_df)
        except Exception as e:
            print(f"Erreur récupération: {e}")

        current_start = current_end + timedelta(days=1)

    # Vérifier si on a récupéré des articles
    if not all_articles:
        # print("Aucun article trouvé pour la période.")
        return pd.DataFrame(columns=['date', 'sentiment'])

    df = pd.concat(all_articles, ignore_index=True)

    # Parsing date GDELT
    def convertir_date_gdelt(date_str):
        try:
            # Format AAAAMMJJTHHMMSSZ ou AAAAMMJJ
            date_part = date_str.split('T')[0]
            return datetime.strptime(date_part, "%Y%m%d").date()
        except:
            return pd.NaT

    df['date'] = df['seendate'].apply(convertir_date_gdelt)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])

    # Calculer sentiment pour chaque article
    def calculer_sentiment(texte):
        if isinstance(texte, str) and texte.strip():
            return sid.polarity_scores(texte)['compound']
        return 0

    df['sentiment'] = df['title'].apply(calculer_sentiment)

    # Agréger par jour : score moyen par jour
    df_daily = df.groupby('date')['sentiment'].mean().reset_index()

    return df_daily

start_date = eur_usd.iloc[0].name.strftime("%Y-%m-%d")
end_date = eur_usd.iloc[-1].name.strftime("%Y-%m-%d")

# print(f"Récupération sentiment GDELT de {start_date} à {end_date}")

df_sentiment = recuperer_sentiment_gdelt(
    start_date=start_date,
    end_date=end_date,
    keyword="EUR/USD",
    language='eng',
    chunk_days=30,
    num_records=250
)

df_sentiment = df_sentiment.groupby('date')['sentiment'].mean()
df_sentiment.index = pd.to_datetime(df_sentiment.index)

# print(df_sentiment.head(10))

# --- Fusion des données de sentiment avec les données EUR/USD ---

eur_usd = eur_usd.merge(df_sentiment.rename("sentiment"), left_index=True, right_index=True, how='left')

# Remplir les NaN si aucun article
eur_usd['sentiment'] = eur_usd['sentiment'].fillna(0)

# print(eur_usd.head(10))
# print(eur_usd.isna().sum())

# --- Variables PCA & Clustering ---

# Copie sécurisée
df = eur_usd.dropna().copy()

# Sélection des features quantitatives 
# On exclut les colonnes non numériques ou non pertinentes
features = df.drop(columns=['target'], errors='ignore').select_dtypes(include=[np.number])

# Standardisation 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA (réduction de dimension) 
# Garde assez de composantes pour expliquer 90 % de la variance
pca = PCA(n_components=0.9, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Créer un DataFrame avec les composantes principales
df_pca = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
    index=df.index
)

# KMeans (clustering sur l’espace PCA) 
# Trouve des structures cachées dans les données
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_pca)

# Fusion finale 
df_extended = pd.concat([df, df_pca], axis=1)

# Résumé 
# print(f"Variance expliquée par PCA : {pca.explained_variance_ratio_.sum():.2%}")
# print(f"Nombre de composantes PCA retenues : {pca.n_components_}")
# print(f"Forme finale du DataFrame : {df_extended.shape}")
# print(df_extended.iloc[0])
# print(df_extended.iloc[-1])
# print(df_extended.columns)

# ---- Variables macro ----

start_date = df_extended.index[0].strftime("%Y-%m-%d")
end_date   = df_extended.index[-1].strftime("%Y-%m-%d")

# print(f"Récupération variables macro de {start_date} à {end_date}")

infl_eur = taceconomics.getdata(
    f"EUROSTAT/EI_CPHI_M_CP-HI00_NSA_HICP2015/EUZ?collapse=D&transform=growth_yoy&start_date={start_date}"
)
infl_us = taceconomics.getdata(
    f"FRED/CPIAUCSL/USA?collapse=D&transform=growth_yoy&start_date={start_date}"
)
ti_eur = taceconomics.getdata(
    f"ECB/FM_D_EUR_4F_KR_DFR_LEV/EUZ?collapse=D&collapse_mode=end_of_period&start_date={start_date}"
)
ti_us = taceconomics.getdata(
    f"DS/USPRATE./WLD?collapse=D&start_date={start_date}"
)

infl_eur = infl_eur.squeeze()
infl_us  = infl_us.squeeze()
ti_eur   = ti_eur.squeeze()
ti_us    = ti_us.squeeze()

for s in [infl_eur, infl_us, ti_eur, ti_us]:
    s.index = pd.to_datetime(s.index).tz_localize(None)

df_macro = pd.DataFrame({
    "inflation_eur": infl_eur,
    "inflation_us": infl_us,
    "interest_rate_eur": ti_eur,
    "interest_rate_us": ti_us,
})


if df_macro.index[-1] < pd.to_datetime(end_date):
    full_index = pd.date_range(start=df_macro.index[0], end=end_date, freq="D")
    df_macro = df_macro.reindex(full_index).ffill()

# print("Macro étendue :")
# print(df_macro.index[0], "→", df_macro.index[-1])
# print(df_macro.head(3))
# print(df_macro.tail(3))

df_final = pd.concat([df_extended, df_macro], axis=1)
df_final = df_final.ffill()

# print(df_final.tail(10))
# print(df_final.columns)
# print(df_final.iloc[0])

#--- Transformation au format long ---

df_long = df_final.stack().reset_index()

# Renommer les colonnes
df_long.columns = ['date', 'indicateur', 'value']

# Trier par date et indicateur
df_long = df_long.sort_values(['date', 'indicateur']).reset_index(drop=True)

print(df_long.head())
print(f"Shape: {df_long.shape}")

# --- Export sur base SQL via conteneur Docker ---

import pandas as pd
from sqlalchemy import create_engine

# Paramètres de connexion
db_user = 'postgres'
db_password = 'mysecretpassword'
db_host = 'localhost'  
db_port = 5432
db_name = 'postgres'    

# Création de l'engine SQLAlchemy
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Export vers SQL
df_long.to_sql('ma_table', engine, if_exists='replace', index=False)