# fx-shock-prediction
Prédiction de chocs sur le taux de change EUR/USD à un horizon de deux semaines via indicateurs financiers et modèles de machine learning.

## Contexte du projet
Ce projet a été réalisé dans le cadre du cours **Applied Big Data Analytics in Finance du Master II Finance Data donné par Didier Liron (2024-2025)**.  
L’objectif du projet est de prédire la probabilité d’un choc sur une variable financière (ici le taux de change EUR/USD) à un horizon de deux semaines.

## Objectif
Détecter à l’avance les mouvements extrêmes du taux EUR/USD en utilisant des indicateurs techniques, macroéconomiques et statistiques, afin d’évaluer la capacité des modèles de machine learning à anticiper des événements rares sur les marchés de change.

## Méthodologie
### 1. Création de la base de données :  
   - Récupération des données quotidiennes du taux EUR/USD grâce au Datalab de TAC Economics.
   - Définition de la variable cible binaire : présence (1) ou absence (0) d’un choc à 2 semaines.
   - Calcul d’indicateurs chartistes : RSI, bandes de Bollinger, etc.  
   - Création d’une variable de sentiment basée sur des articles issus de la base GDELT, donnant un score compris entre -1 (négatif) et 1 (positif) à l’aide du modèle VADER.  
   - Construction de variables explicatives additionnelles issues de méthodes de réduction de dimension (PCA) et de clustering (k-means).
   - Ajout de variables Macro issues du Datalab de TAC Economics
   - Définition de la variable cible binaire : présence (1) ou absence (0) d’un choc à 2 semaines.

 ### 2. Modélisation :  
   - Entraînement d'un modèle XGboost
   - Découpage temporel 80% train, 20% de test.  

### 3. Optimisation et évaluation :  
   - Optimisation des hyperparamètres.  
   - Analyse des performances à l’aide des courbes ROC, matrices de confusion et de l’indice de Gini.

### 4. Docker
   - Exportation de la base dans une base mysql sous Docker au format panel
   - Permet à tout le monde de pouvoir importer la base de données afin de pourvoir appliquer le modèle de ML.
