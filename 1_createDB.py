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
eur_usd["cible"] = eur_usd.apply(
    lambda row: 1 if (row["rendement_10j"] > row["seuil_haut"]) or (row["rendement_10j"] < row["seuil_bas"]) else 0,
    axis=1
)

# --- Indicateurs chartistes pour visualisation ---

# Moyennes mobiles
eur_usd["moyenne_mobile_7j"] = eur_usd["eur_usd"].rolling(window=7).mean()
eur_usd["moyenne_mobile_21j"] = eur_usd["eur_usd"].rolling(window=21).mean()

# Bandes de Bollinger
eur_usd["bollinger_moyenne"] = eur_usd["eur_usd"].rolling(window=20).mean()
eur_usd["bollinger_haut"] = eur_usd["bollinger_moyenne"] + 2 * eur_usd["eur_usd"].rolling(window=20).std()
eur_usd["bollinger_bas"] = eur_usd["bollinger_moyenne"] - 2 * eur_usd["eur_usd"].rolling(window=20).std()

print(eur_usd.head())


# RSI (Relative Strength Index)
def calculer_rsi(series, window=14):
    delta = series.diff()
    gains = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    pertes = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gains / pertes
    rsi = 100 - (100 / (1 + rs))
    return rsi

eur_usd["rsi_14j"] = calculer_rsi(eur_usd["eur_usd"])

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
    print(df.head())


# --- Utilisation ---
start_date = '2023-01-01'
end_date = '2025-09-03'

# Récupérer tous les articles
articles_df = recuperer_tous_articles_gdelt(start_date, end_date, keyword="EUR/USD")

if not articles_df.empty:
    # Calculer le sentiment moyen par jour
    sentiment_quotidien = articles_df.groupby('date')['sentiment'].mean().reset_index()

    # Fusionner avec ton DataFrame eur_usd
    eur_usd_indexed = eur_usd.reset_index()
    eur_usd_indexed['timestamp'] = pd.to_datetime(eur_usd_indexed['timestamp']).dt.normalize()
 # Normaliser la date

    eur_usd_with_sentiment = eur_usd_indexed.merge(
       sentiment_quotidien,
        on='timestamp',
        how='left'
    ).set_index('index')

    eur_usd_with_sentiment.rename(columns={'sentiment': 'sentiment_vader'}, inplace=True)
    eur_usd_with_sentiment['sentiment_vader'] = eur_usd_with_sentiment['sentiment_vader'].fillna(0)

    # Mettre à jour ton DataFrame eur_usd
    eur_usd = eur_usd_with_sentiment

    # Afficher les dernières valeurs
    print(eur_usd[['eur_usd', 'sentiment_vader']].tail(10))

# --- Variables exogènes économiques ---

# def charger_serie(nom_colonne, url):
#     """Charge une série via l'API TAC, la nettoie et renomme sa colonne"""
#     df = taceconomics.getdata(url)
#     if df is not None:
#         df = df.dropna()
#         df.columns = [nom_colonne]
#         df.index = pd.to_datetime(df.index)
#         return df
#     else:
#         print(f"Erreur : données non chargées pour {nom_colonne}")
#         return None

# # Dictionnaire des séries exogènes à importer
# series_info = {
#     "infl_USA":   f"FRED/CPIAUCSL/USA?collapse=M&transform=growth_yoy&start_date={start_date}",
#     "infl_EUZ":   f"EUROSTAT/EI_CPHI_M_CP-HI00_NSA_HICP2015/EUZ?collapse=M&transform=growth_yoy&start_date={start_date}",
#     "prate_USA":  f"DS/USPRATE./WLD?collapse=M&start_date={start_date}",
#     "prate_EUZ":  f"ECB/FM_D_EUR_4F_KR_DFR_LEV/EUZ?collapse=M&collapse_mode=end_of_period&start_date={start_date}",
#     "bond10_USA": f"DS/TRUS10T_RY/WLD?collapse=M&start_date={start_date}",
#     "bond10_EUZ": f"DS/TRBD10T_RY/WLD?collapse=M&start_date={start_date}",
#     "tgdp_USA":   f"FRED/GDPC1/USA?collapse=M&transform=growth_yoy&start_date={start_date}",
#     "tgdp_EUZ":   f"EUROSTAT/NAMQ_10_GDP_B1GQ_SCA_CLV15_MEUR/EUZ?collapse=M&transform=growth_yoy&start_date={start_date}"
# }

# # Chargement et fusion des séries dans une seule table
# dataframes = []
# for nom, url in series_info.items():
#     df = charger_serie(nom, url)
#     if df is not None:
#         dataframes.append(df)

# # Fusion de toutes les séries exogènes par date
# df_exog = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dataframes)

# # Fusion finale : taux de change + exogènes
# df_final = pd.merge(eur_usd, df_exog, left_index=True, right_index=True, how='outer')
# df_final = df_final.ffill().sort_index()  # Remplissage des valeurs manquantes et tri
