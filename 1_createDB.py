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
