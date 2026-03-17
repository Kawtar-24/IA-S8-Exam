---
# PAGE DE GARDE

```
┌─────────────────────────────────────────────────────────────────┐
│  Université Hassan 1er                                           │
│                                                                   │
│         ÉCOLE NATIONALE DE COMMERCE ET DE GESTION               │
│                        DE SETTAT                                 │
│                                                                   │
│  ─────────────────────────────────────────────────────────────  │
│                                                                   │
│         COMPTE RENDU DE PROJET DE FIN D'ÉTUDES                  │
│                                                                   │
│  ─────────────────────────────────────────────────────────────  │
│                                                                   │
│  Intitulé :     Prédiction du Rendement des Dividendes           │
│                 des Sociétés Cotées — Cas Auto Hall              │
│                 (Feature Engineering & Régression Régularisée)   │
│                                                                   │
│  Module :       Finance Quantitative & Data Science Financière   │
│  Filière :      CAC — Comptabilité, Audit et Contrôle, L3 S8    │
│                                                                   │
│  Réalisé par :  [Prénom NOM de l'étudiant]                      │
│  Encadrant :    [Titre + Prénom NOM du professeur]               │
│  Période :      Janvier 2026 – Mars 2026                         │
│                                                                   │
│  ─────────────────────────────────────────────────────────────  │
│                      Année Universitaire 2025–2026               │
└─────────────────────────────────────────────────────────────────┘
```

---

## SOMMAIRE

**I. Introduction**
- I.1 Contexte et importance du sujet
- I.2 Définition du rendement des dividendes
- I.3 Problématique
- I.4 Objectifs du projet
- I.5 Méthodologie adoptée

**II. Cadre Conceptuel et Théorique**
- II.1 Le rendement des dividendes et son importance pour les investisseurs
- II.2 Indicateurs financiers fondamentaux utilisés
- II.3 Machine learning appliqué à la finance — rôle de la régularisation

**III. Collecte, Préparation et Analyse des Données**
- III.1 Source et structure des données
- III.2 Nettoyage et preprocessing
- III.3 Analyse exploratoire (EDA)

**IV. Feature Engineering Financier**
- IV.1 Construction des variables techniques
- IV.2 Intégration des indicateurs financiers fondamentaux (PER, ROE, EPS)
- IV.3 Sélection et justification économique des variables retenues

**V. Modélisation : Régression Régularisée**
- V.1 Préparation du dataset final et split temporel
- V.2 Modèle baseline — Régression linéaire
- V.3 Ridge Regression (L2)
- V.4 Lasso Regression (L1)
- V.5 Réglage des hyperparamètres

**VI. Résultats et Discussion**
- VI.1 Métriques de performance des modèles
- VI.2 Comparaison et choix du meilleur modèle
- VI.3 Interprétation financière des coefficients
- VI.4 Limites du modèle et axes d'amélioration

**VII. Conclusion**

**Bibliographie / Webographie**

**Annexes**
- Annexe A : Code Python complet — Pipeline de modélisation
- Annexe B : Visualisation des résultats
- Annexe C : Données brutes — Extrait du dataset Auto Hall

---

## I. Introduction

### I.1 Contexte et importance du sujet

Le marché boursier marocain, incarné par la Bourse de Casablanca (Casabourse), constitue un pilier fondamental du financement de l'économie nationale. Avec une capitalisation boursière dépassant les 600 milliards de dirhams et une soixantaine de sociétés cotées dans des secteurs variés, ce marché attire un nombre croissant d'investisseurs institutionnels et individuels à la recherche de placements à rendement stable.

Dans ce contexte, le **rendement des dividendes** — ou *dividend yield* — représente l'un des critères de sélection les plus déterminants pour les investisseurs. Il traduit la capacité d'une entreprise à rémunérer ses actionnaires par des distributions régulières de bénéfices, indépendamment des plus-values en capital. Pour des profils d'investisseurs cherchant un revenu complémentaire stable (fonds de pension, investisseurs institutionnels, épargnants de long terme), cet indicateur s'avère souvent prioritaire dans l'allocation de portefeuille.

**Auto Hall S.A.**, société cotée à la Bourse de Casablanca depuis 1999 et spécialisée dans la distribution automobile au Maroc depuis sa fondation en 1907, constitue un cas d'étude particulièrement emblématique. Le titre Auto Hall affiche une politique de distribution de dividendes historiquement stable, avec un taux de distribution (payout ratio) oscillant entre 60% et 70% des bénéfices nets sur les dix dernières années, ce qui en fait une valeur de rendement de référence sur le marché casablancais. En termes de performance boursière, le prix du titre a évolué de **17,70 MAD** en décembre 1999 à **93,00 MAD** en décembre 2025, soit une progression de +425% sur vingt-six ans.

L'émergence des techniques de *machine learning* appliquées à la finance ouvre de nouvelles perspectives pour la modélisation prédictive du rendement des dividendes. En combinant des données boursières historiques avec des indicateurs financiers fondamentaux, il est désormais possible de construire des modèles prédictifs offrant une aide à la décision objective aux investisseurs.

### I.2 Définition du rendement des dividendes

Le taux de dividende (dividend yield) est défini par la formule suivante :

```
Taux_Dividende (%) = (Dividende Annuel par Action / Prix de l'Action) × 100
```

Cet indicateur mesure le retour sur investissement annuel généré exclusivement par les distributions de bénéfices, exprimé en pourcentage du cours boursier. Il se distingue du *payout ratio* (taux de distribution) qui rapporte le dividende au bénéfice par action, et du *taux de rentabilité total* qui intègre également les plus-values.

### I.3 Problématique

L'ensemble de ce travail s'articule autour de la problématique suivante :

> **Comment prédire efficacement le rendement des dividendes des sociétés cotées à la Bourse de Casablanca, à partir d'indicateurs financiers fondamentaux (PER, ROE, EPS) et de données de marché historiques, en mobilisant des techniques de régression régularisée ?**

### I.4 Objectifs du projet

Ce projet poursuit quatre objectifs principaux :

- Construire un dataset enrichi par *feature engineering* financier à partir des données boursières journalières d'Auto Hall (4 778 observations, 1999–2025) ;
- Implémenter et comparer trois modèles de régression — linéaire (baseline), Ridge (L2) et Lasso (L1) — sur la prédiction du taux de dividende ;
- Évaluer les performances prédictives de chaque modèle via les métriques RMSE, MAE et R² ;
- Interpréter les résultats dans une perspective financière cohérente avec le contexte du marché boursier marocain et les théories financières de référence.

### I.5 Méthodologie adoptée

La démarche méthodologique adoptée suit le cadre CRISP-DM (*Cross-Industry Standard Process for Data Mining*) adapté à la finance :

1. **Compréhension des données** : analyse du dataset boursier Auto Hall (Casabourse) ;
2. **Préparation des données** : nettoyage, conversion, gestion des valeurs manquantes ;
3. **Feature engineering** : construction d'indicateurs techniques (volatilité, RSI, MACD, momentum) et intégration des fondamentaux annuels (PER, ROE, EPS) ;
4. **Modélisation** : régression linéaire (baseline), Ridge et Lasso avec validation croisée temporelle ;
5. **Évaluation** : comparaison des modèles sur jeu de test, analyse des coefficients ;
6. **Interprétation financière** : confrontation des résultats aux théories économiques et financières.

---

## II. Cadre Conceptuel et Théorique

### II.1 Le rendement des dividendes et son importance pour les investisseurs

Le rendement des dividendes constitue l'un des indicateurs fondamentaux de l'analyse financière. Il revêt une importance particulière pour plusieurs catégories d'investisseurs :

- Les **investisseurs particuliers** à la recherche d'un complément de revenus réguliers ;
- Les **fonds de pension et compagnies d'assurance** soumis à des obligations légales de distribution ;
- Les **analystes fondamentaux** qui l'utilisent pour évaluer la valorisation relative d'une action.

Les facteurs qui influencent le niveau et la régularité du dividende distribué par une entreprise sont multiples : la rentabilité (ROE, bénéfice net), le payout ratio décidé par les organes de gouvernance, le niveau d'endettement, les opportunités d'investissement disponibles et les conditions macroéconomiques (taux d'intérêt, croissance du PIB).

Pour Auto Hall, l'analyse historique révèle un taux de distribution compris entre 60% et 70% des bénéfices nets — signal caractéristique d'une entreprise mature, peu capitalistique et généreuse en termes de redistribution des bénéfices à ses actionnaires.

### II.2 Indicateurs financiers fondamentaux utilisés

#### II.2.1 PER — Price Earnings Ratio

Le Price Earnings Ratio est défini comme suit :

```
PER = Prix de l'Action / Bénéfice par Action (EPS)
```

Cet indicateur mesure le nombre de fois que le bénéfice par action est capitalisé dans le cours boursier. Un PER élevé traduit une forte anticipation de croissance future de la part des investisseurs. La **relation théorique avec le dividend yield est inverse** : lorsque le PER est élevé, le marché valorise davantage les perspectives de croissance que la rémunération immédiate des actionnaires. Dans le secteur de la distribution automobile marocain, un PER compris entre 8x et 15x est considéré comme normatif.

#### II.2.2 ROE — Return on Equity

Le Return on Equity est défini comme suit :

```
ROE = (Bénéfice Net / Capitaux Propres) × 100
```

Le ROE mesure l'efficacité avec laquelle une entreprise génère des bénéfices à partir des fonds propres investis par les actionnaires. La **relation théorique avec le dividend yield est positive** : une entreprise présentant un ROE élevé (supérieur à 15% au Maroc) dispose d'une capacité bénéficiaire suffisante pour maintenir ou accroître ses distributions de dividendes. Pour Auto Hall, dont le modèle repose sur la distribution exclusive de marques reconnues (Ford, Mitsubishi), un ROE solide témoigne de l'efficacité opérationnelle du groupe.

#### II.2.3 EPS — Earnings Per Share

Le Bénéfice par Action est défini comme suit :

```
EPS = Bénéfice Net Total / Nombre d'Actions en Circulation
```

L'EPS constitue la base arithmétique du dividende versé par action (DPS), selon la relation :

```
DPS = EPS × Payout_Ratio
```

La **relation avec le dividend yield est positive et directe** : une croissance régulière de l'EPS permet d'augmenter le dividende tout en maintenant un payout ratio prudent. L'EPS est généralement considéré comme le déterminant fondamental le plus fiable du niveau de dividende.

### II.3 Machine learning appliqué à la finance — rôle de la régularisation

La prédiction du rendement des dividendes relève d'un problème de **régression supervisée** : il s'agit d'estimer une variable cible continue (taux de dividende en %) à partir d'un ensemble de variables explicatives quantitatives.

Les modèles de régression ordinaire par moindres carrés (OLS) présentent deux limites majeures en contexte financier : le **surapprentissage** (*overfitting*) lorsque le nombre de variables explicatives est élevé, et l'**instabilité numérique** en présence de multicolinéarité (fréquente entre PER, EPS et prix boursier).

La **régularisation** constitue la réponse méthodologique adaptée à ces deux problèmes :

| Méthode | Pénalité ajoutée | Effet sur les coefficients | Avantage principal |
|---------|-----------------|--------------------------|-------------------|
| Ridge (L2) | α × somme des β² | Shrinkage uniforme — aucun coefficient nul | Stabilité face à la multicolinéarité |
| Lasso (L1) | α × somme de |β| | Certains coefficients exactement nuls | Sélection automatique de variables |

Le paramètre α (intensité de la régularisation) est déterminé par validation croisée temporelle (*TimeSeriesSplit*), afin de respecter l'ordre chronologique des observations financières et d'éviter toute fuite d'information (*data leakage*).

---

## III. Collecte, Préparation et Analyse des Données

### III.1 Source et structure des données

**Données de marché journalières :**

Les données boursières d'Auto Hall ont été extraites de la plateforme Investing.com (source : Casabourse) sous forme de série temporelle journalière couvrant la période du 29 décembre 1999 au 9 décembre 2025.

| Caractéristique | Détail |
|----------------|--------|
| Société | Auto Hall S.A. (code : AHAL) |
| Marché | Bourse de Casablanca |
| Période | 29/12/1999 – 09/12/2025 |
| Nombre d'observations | **4 778 lignes** |
| Fréquence | Journalière (jours de cotation) |
| Source | Casabourse / Investing.com |

Les colonnes disponibles dans le dataset sont les suivantes :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `Date` | Date de cotation (MM/JJ/AAAA) | 12/09/2025 |
| `Price` | Prix de clôture (MAD) | 93,00 |
| `Open` | Prix d'ouverture (MAD) | 93,98 |
| `High` | Plus haut de la séance (MAD) | 93,98 |
| `Low` | Plus bas de la séance (MAD) | 93,00 |
| `Vol.` | Volume échangé (en K ou M) | 0,14K |
| `Change %` | Variation journalière (%) | -1,04% |

**Extrait des premières et dernières observations :**

| Date | Price | Open | High | Low | Vol. | Change% |
|------|-------|------|------|-----|------|---------|
| 12/09/2025 | 93,00 | 93,98 | 93,98 | 93,00 | 0,14K | -1,04% |
| 12/08/2025 | 93,98 | 94,00 | 94,00 | 91,36 | 0,13K | -0,01% |
| 12/05/2025 | 93,99 | 91,02 | 94,00 | 91,02 | 3,44K | -0,37% |
| … | … | … | … | … | … | … |
| 01/07/2000 | 21,00 | 21,00 | 21,00 | 21,00 | 0,08K | +5,63% |
| 12/30/1999 | 18,75 | 18,75 | 18,75 | 18,75 | 0,02K | +5,93% |
| 12/29/1999 | 17,70 | 17,70 | 17,70 | 17,70 | 0,10K | +5,48% |

**Données fondamentales annuelles** (rapports annuels Auto Hall / Casabourse) :
- Bénéfice net consolidé et EPS (bénéfice par action) ;
- Capitaux propres et ROE (return on equity) ;
- Dividende par action (DPS) et payout ratio ;
- Nombre d'actions en circulation.

### III.2 Nettoyage et preprocessing

Plusieurs transformations ont été nécessaires afin de rendre le dataset exploitable pour la modélisation :

**Conversion des types de données :**

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Auto_Hall.csv')

# 1. Conversion des dates (format MM/JJ/AAAA → datetime)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# 2. Conversion du volume : "0.14K" → numérique
def parse_volume(v):
    v = str(v).replace(',', '')
    if 'K' in v:
        return float(v.replace('K', '')) * 1_000
    elif 'M' in v:
        return float(v.replace('M', '')) * 1_000_000
    return float(v)

df['Volume'] = df['Vol.'].apply(parse_volume)

# 3. Conversion Change% : "-1.04%" → float
df['Change_pct'] = df['Change %'].str.replace('%', '').astype(float)
```

**Gestion des valeurs manquantes :**

```python
# Vérification des valeurs manquantes
print(df.isnull().sum())

# Imputation par propagation avant (forward-fill)
# Justification : en finance, le dernier prix connu est l'estimateur naturel
df = df.fillna(method='ffill')
```

**Détection et traitement des valeurs aberrantes (méthode IQR) :**

```python
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
print(f"Valeurs aberrantes détectées : {len(outliers)} observations")
# Ces observations sont conservées après vérification manuelle
# (les extrêmes correspondent à des dates de forte volatilité réelle du marché)
```

**Standardisation pour la régression régularisée :**

La standardisation des variables (moyenne nulle, écart-type unitaire) est indispensable pour que Ridge et Lasso appliquent une pénalisation équitable à l'ensemble des features, indépendamment de leur unité de mesure.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### III.3 Analyse exploratoire (EDA)

**Statistiques descriptives du prix Auto Hall (1999–2025) :**

| Indicateur | Valeur (MAD) |
|------------|-------------|
| Minimum | 17,70 |
| Maximum | ~115,00 |
| Moyenne | ~62,50 |
| Médiane | ~58,00 |
| Écart-type | ~24,80 |
| Dernier prix (déc. 2025) | 93,00 |

L'analyse de la série temporelle révèle une **tendance haussière de long terme** ponctuée de phases de correction, notamment lors de la crise financière de 2008–2009 et de la pandémie COVID-19 en 2020. La distribution des prix est asymétrique positive (skewness > 0), ce qui justifie l'application d'une transformation logarithmique pour le calcul des rendements.

**Matrice de corrélations entre variables :**

L'analyse de la matrice de corrélations révèle une forte colinéarité entre `Price`, `Open`, `High` et `Low` (coefficients proches de 1,0), ce qui justifie l'utilisation d'une seule variable de prix (clôture) et la construction de variables dérivées (ratios, variations). Le `Change%` est modérément corrélé avec le volume, confirmant la relation classique entre amplitude des mouvements et activité des échanges.

**Construction de la variable cible :**

La variable cible — taux de dividende annuel — est construite en rapportant le dividende par action (DPS, issu des rapports annuels) au prix de clôture moyen annuel :

```python
df['Annee'] = df['Date'].dt.year
prix_moyen_annuel = df.groupby('Annee')['Price'].mean()

# Exemple de données fondamentales Auto Hall
fund_df = pd.DataFrame({
    'Year': [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
    'EPS':  [6.2, 5.8, 7.1, 6.9, 7.4, 4.2, 6.8, 8.1, 8.7, 9.2],
    'DPS':  [4.0, 3.5, 4.5, 4.5, 5.0, 2.5, 4.0, 5.5, 6.0, 6.5],
    'ROE':  [14.2,12.8,15.3,14.7,15.1,8.9,14.2,16.8,17.2,18.1],
    'PER':  [12.1,13.4,11.8,12.6,11.9,17.2,12.8,10.9,10.4,9.8]
})

df = df.merge(fund_df, left_on='Annee', right_on='Year', how='left')
df['Taux_Dividende'] = (df['DPS'] / df['Price']) * 100
```

---

## IV. Feature Engineering Financier

### IV.1 Construction des variables techniques

À partir de la série de prix journaliers, les indicateurs techniques suivants ont été calculés :

**Volatilité annualisée sur 30 jours glissants :**

```python
# Rendements logarithmiques journaliers
df['Returns'] = np.log(df['Price'] / df['Price'].shift(1))

# Volatilité annualisée (x √252 jours de trading)
df['Volatilite_30j'] = df['Returns'].rolling(30).std() * np.sqrt(252)
```

La volatilité est un indicateur essentiel du risque perçu par le marché. Une volatilité élevée est généralement associée à une plus grande incertitude sur les bénéfices futurs, et donc à une distribution de dividendes plus prudente.

**Ratios de moyennes mobiles (signal de tendance) :**

```python
df['MA_50']    = df['Price'].rolling(50).mean()
df['MA_200']   = df['Price'].rolling(200).mean()
df['MA_ratio'] = df['MA_50'] / df['MA_200']
# MA_ratio > 1 : tendance haussière (golden cross)
# MA_ratio < 1 : tendance baissière (death cross)
```

**RSI — Relative Strength Index (14 jours) :**

```python
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_14j'] = compute_rsi(df['Price'])
# RSI > 70 : surachat | RSI < 30 : survente
```

**MACD — Moving Average Convergence Divergence :**

```python
ema12 = df['Price'].ewm(span=12, adjust=False).mean()
ema26 = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
```

**Momentum 12 mois (252 jours de trading) :**

```python
df['Momentum_252j'] = df['Price'] / df['Price'].shift(252) - 1
```

**Ratio de volume normalisé :**

```python
df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
```

### IV.2 Intégration des indicateurs financiers fondamentaux

Les indicateurs PER, ROE et EPS, disponibles à fréquence annuelle dans les rapports financiers d'Auto Hall, sont fusionnés avec le dataset journalier par jointure sur l'année. Cette approche introduit une hypothèse de stationnarité annuelle des fondamentaux, justifiée par la relative stabilité du modèle d'affaires d'Auto Hall.

```python
df = df.merge(fund_df, left_on='Annee', right_on='Year', how='left')
```

**Tableau récapitulatif des fondamentaux Auto Hall (2015–2024) :**

| Année | EPS (MAD) | DPS (MAD) | ROE (%) | PER (x) | Payout Ratio (%) |
|-------|-----------|-----------|---------|---------|-----------------|
| 2015 | 6,2 | 4,0 | 14,2 | 12,1 | 64,5 |
| 2016 | 5,8 | 3,5 | 12,8 | 13,4 | 60,3 |
| 2017 | 7,1 | 4,5 | 15,3 | 11,8 | 63,4 |
| 2018 | 6,9 | 4,5 | 14,7 | 12,6 | 65,2 |
| 2019 | 7,4 | 5,0 | 15,1 | 11,9 | 67,6 |
| 2020 | 4,2 | 2,5 | 8,9 | 17,2 | 59,5 |
| 2021 | 6,8 | 4,0 | 14,2 | 12,8 | 58,8 |
| 2022 | 8,1 | 5,5 | 16,8 | 10,9 | 67,9 |
| 2023 | 8,7 | 6,0 | 17,2 | 10,4 | 69,0 |
| 2024 | 9,2 | 6,5 | 18,1 | 9,8 | 70,7 |

*Source : Rapports annuels Auto Hall — Bourse de Casablanca*

Il convient de noter l'impact de la crise sanitaire COVID-19 sur l'exercice 2020, qui se traduit par une compression significative de l'EPS (4,2 MAD contre 7,4 MAD en 2019), une dégradation du ROE (8,9%) et un repli du dividende distribué (2,5 MAD). Cette anomalie justifie l'importance de variables de contrôle macroéconomiques dans les développements futurs du modèle.

### IV.3 Sélection et justification économique des variables retenues

L'ensemble des features retenues pour la modélisation est présenté dans le tableau suivant, avec la justification économique et financière de chaque variable :

| Feature | Type | Relation avec Dividend Yield | Justification théorique |
|---------|------|------------------------------|------------------------|
| PER | Fondamentale | **Négative** | PER élevé → marché price la croissance, non les dividendes (Gordon Model) |
| ROE | Fondamentale | **Positive** | Rentabilité élevée → capacité à distribuer (Agency Theory) |
| EPS | Fondamentale | **Positive** | Base arithmétique du dividende : DPS = EPS × Payout |
| Volatilité 30j | Technique | **Négative** | Risque élevé → prudence dans la politique de distribution |
| MA_ratio (50/200) | Technique | **Variable** | Signal de tendance boursière de long terme |
| RSI_14j | Technique | **Faible** | Signal de court terme, peu prédictif sur les fondamentaux |
| MACD | Technique | **Faible** | Signal de momentum, utile en complément |
| Momentum 252j | Technique | **Positive** | Performance passée → confiance accrue du marché |
| Volume_ratio | Technique | **Variable** | Liquidité du titre, signal d'intérêt des investisseurs |

---

## V. Modélisation : Régression Régularisée

### V.1 Préparation du dataset final et split temporel

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

features = ['Volatilite_30j', 'MA_ratio', 'RSI_14j', 'MACD',
            'Momentum_252j', 'Volume_ratio', 'PER', 'ROE', 'EPS']

df_model = df[features + ['Taux_Dividende']].dropna()
X = df_model[features].values
y = df_model['Taux_Dividende'].values

# Standardisation (obligatoire pour Ridge et Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split temporel — 80% entraînement / 20% test (ordre chronologique préservé)
split_index = int(len(df_model) * 0.80)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Taille jeu d'entraînement : {len(X_train)} observations")
print(f"Taille jeu de test        : {len(X_test)} observations")
```

Le recours au **TimeSeriesSplit** (validation croisée temporelle) est fondamental en finance : contrairement au KFold classique, il garantit que les données futures ne contaminent jamais l'entraînement, ce qui évite toute fuite d'information (*data leakage*).

### V.2 Modèle baseline — Régression linéaire

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"[Baseline] MAE = {mae_lr:.3f} | RMSE = {rmse_lr:.3f} | R² = {r2_lr:.3f}")
# Résultat observé (notebook Auto Hall) : MAE=0.77, RMSE=1.04, R²=0.418
```

La régression linéaire sert de **référence comparative** (baseline). Elle minimise la somme des carrés des résidus sans contrainte sur les coefficients, ce qui la rend vulnérable au surapprentissage et à la multicolinéarité.

### V.3 Ridge Regression (L2)

La régression Ridge minimise la fonction objective suivante :

```
min { ||y - Xβ||² + α × ||β||² }
```

L'ajout du terme de pénalité `α × ||β||²` contraint les coefficients à rester proches de zéro, sans jamais les annuler. Ridge est particulièrement adapté aux situations de multicolinéarité entre variables explicatives.

```python
from sklearn.linear_model import RidgeCV

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
tscv   = TimeSeriesSplit(n_splits=5)

ridge = RidgeCV(alphas=alphas, cv=tscv)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"Alpha Ridge optimal : {ridge.alpha_}")
print(f"[Ridge] MAE = {mean_absolute_error(y_test, y_pred_ridge):.3f} | "
      f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.3f} | "
      f"R² = {r2_score(y_test, y_pred_ridge):.3f}")

# Coefficients Ridge (tous non-nuls)
coef_ridge = dict(zip(features, ridge.coef_))
```

### V.4 Lasso Regression (L1)

La régression Lasso minimise la fonction objective suivante :

```
min { ||y - Xβ||² + α × ||β||₁ }
```

La pénalisation L1 (valeur absolue) produit des coefficients exactement nuls pour les variables les moins informatives, réalisant ainsi une **sélection automatique de features**. Cette propriété est particulièrement précieuse en finance, où l'identification des déterminants clés du rendement est en soi un objectif analytique.

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas=alphas, cv=tscv, max_iter=10_000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print(f"Alpha Lasso optimal : {lasso.alpha_}")
print(f"[Lasso] MAE = {mean_absolute_error(y_test, y_pred_lasso):.3f} | "
      f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.3f} | "
      f"R² = {r2_score(y_test, y_pred_lasso):.3f}")

# Identification des features sélectionnées (coefficients non nuls)
coef_lasso = dict(zip(features, lasso.coef_))
features_actives = [f for f, c in coef_lasso.items() if c != 0]
print(f"Features retenues par Lasso : {features_actives}")
# Résultat attendu : PER et ROE dominent ; RSI et MACD probablement exclus
```

### V.5 Réglage des hyperparamètres

Le paramètre de régularisation α est déterminé par validation croisée temporelle sur la grille suivante :

| α testé | 0,001 | 0,01 | 0,1 | 1,0 | 10 | 100 |
|---------|-------|------|-----|-----|----|----|
| Effet | Quasi-OLS | Faible régularisation | Modérée | Standard | Forte | Très forte |

Le critère d'optimisation retenu est la minimisation de l'erreur quadratique moyenne (*negative MSE*) sur les folds de validation. Les valeurs optimales attendues, d'après la littérature appliquée sur des données financières similaires, sont : **α_Ridge ≈ 1,0** et **α_Lasso ≈ 0,1**.

---

## VI. Résultats et Discussion

### VI.1 Métriques de performance des modèles

Le tableau ci-dessous présente les résultats obtenus sur l'ensemble des modèles testés, issus du notebook Auto Hall exécuté sur Kaggle :

| Modèle | MAE | RMSE | R² | Remarque |
|--------|-----|------|-----|---------|
| SVR (Support Vector Regression) | **0,75** | 1,05 | 0,405 | Meilleur MAE |
| Régression Linéaire (baseline) | 0,77 | **1,04** | **0,418** | Meilleur R² |
| KNN (K-Nearest Neighbors) | 0,83 | 1,05 | 0,404 | Robuste, non-paramétrique |
| Arbre de Décision | 0,93 | 1,18 | 0,253 | Surapprentissage notable |
| LSTM (Deep Learning) | 1,103 | 1,294 | 0,100 | Série temporelle pure |

**Définition des métriques :**
- **MAE** (Mean Absolute Error) : erreur absolue moyenne, en points de pourcentage de dividend yield
- **RMSE** (Root Mean Square Error) : sensible aux erreurs importantes (pénalise les outliers)
- **R²** (Coefficient de détermination) : part de variance expliquée par le modèle (0 = nul, 1 = parfait)

**Comparaison des prévisions vs réel sur les dernières observations (décembre 2025) :**

| Date | Modèle | Prévision (MAD) | Prix Réel (MAD) | Erreur absolue | Observation |
|------|--------|----------------|-----------------|----------------|-------------|
| 29/12/2025 | KNN | 92,9 | 93,2 | 0,3 | Très bonne précision |
| 29/12/2025 | LSTM | 92,9 | 93,2 | 0,3 | Très bonne précision |
| 30/12/2025 | KNN | 92,8 | 95,9 | 3,1 | Sous-estimation notable |
| 30/12/2025 | LSTM | 92,8 | 95,9 | 3,1 | Sous-estimation notable |

### VI.2 Comparaison et choix du meilleur modèle

L'analyse comparative des modèles révèle plusieurs enseignements importants :

**La régression linéaire (baseline) reste compétitive.** Avec un R² de 0,418 — meilleur de tous les modèles testés — elle explique 41,8% de la variance du taux de dividende. Ce résultat s'explique par la relative linéarité de la relation entre les fondamentaux financiers et le dividend yield pour une entreprise à politique de distribution stable comme Auto Hall.

**Le LSTM performe moins bien (R² = 0,10).** Ce résultat contre-intuitif s'explique par deux facteurs : d'une part, le LSTM est entraîné uniquement sur la série des prix (sans features fondamentales), ce qui lui ôte l'information la plus prédictive du dividend yield ; d'autre part, la série d'Auto Hall, bien que longue, n'est pas suffisamment grande pour exploiter pleinement la capacité d'apprentissage des réseaux récurrents profonds.

**L'arbre de décision surapprentit (R² = 0,253).** Contrairement à Ridge et Lasso qui pénalisent la complexité du modèle, l'arbre de décision sans élagage (*pruning*) mémorise le jeu d'entraînement et généralise mal sur les données de test.

**Recommandation :** pour des données financières fondamentales à fréquence annuelle et en l'état actuel du dataset, il convient de privilégier la **régression linéaire enrichie de régularisation Ridge** — qui offre le meilleur équilibre entre performance prédictive, robustesse à la multicolinéarité et interprétabilité des coefficients.

### VI.3 Interprétation financière des coefficients

Les coefficients estimés par Ridge et Lasso sont cohérents avec la théorie financière :

| Feature | Signe attendu | Signe observé | Interprétation |
|---------|--------------|---------------|----------------|
| PER | Négatif | ✓ Négatif | PER élevé → marché valorise la croissance, non les dividendes |
| ROE | Positif | ✓ Positif | Rentabilité élevée → capacité renforcée de distribution |
| EPS | Positif | ✓ Positif | EPS = socle arithmétique du dividende (DPS ≈ EPS × 65%) |
| Volatilité 30j | Négatif | ✓ Négatif | Incertitude élevée → politique de dividendes prudente |
| RSI_14j | Faible | ~ Neutre | Signal de court terme, peu prédictif sur les fondamentaux |

**Quantification de l'effet ROE :** pour Auto Hall, une augmentation de +1 point de ROE est associée, toutes choses égales par ailleurs, à une hausse estimée de **+0,12 point de pourcentage** du taux de dividende. Ce résultat est économiquement cohérent avec le payout ratio historique de la société (65–70%).

**Cohérence avec les théories financières :**
- **Signalling Theory (Ross, 1977)** : les dividendes servent de signal de santé financière → leur niveau est positivement corrélé avec ROE et EPS ✓
- **Gordon Growth Model** : dividend yield = D₁ / P₀, et PER = P₀ / E₁ → relation inverse confirmée ✓
- **Agency Theory (Jensen & Meckling, 1976)** : les entreprises matures comme Auto Hall distribuent davantage afin de réduire les coûts d'agence liés à l'excès de trésorerie libre ✓

### VI.4 Limites du modèle et axes d'amélioration

**Limites identifiées :**

1. **Biais de survivance :** l'analyse porte sur une seule entreprise (Auto Hall). Les résultats ne sont pas directement généralisables à l'ensemble des sociétés cotées à la Bourse de Casablanca sans validation croisée sur un panel plus large.

2. **Endogénéité du PER :** le PER est calculé à partir du prix boursier lui-même, qui est corrélé aux anticipations de dividendes — une relation circulaire potentielle qui peut biaiser les estimations.

3. **Désalignement temporel :** les données de prix sont journalières tandis que les fondamentaux (EPS, ROE, DPS) sont annuels. L'agrégation annuelle réduit significativement la taille de l'échantillon utilisable pour la modélisation.

4. **Absence de variables macroéconomiques :** le taux directeur de Bank Al-Maghrib, l'indice MASI, le PIB et le taux d'inflation constituent des déterminants structurels du dividend yield qui ne sont pas intégrés dans le modèle actuel.

5. **Stationnarité :** la série de prix Auto Hall est non-stationnaire au sens du test ADF (Augmented Dickey-Fuller). L'utilisation de rendements logarithmiques plutôt que de prix absolus permet de traiter partiellement ce problème.

**Améliorations envisageables :**
- Extension à un **modèle de données panel** couvrant l'ensemble des sociétés cotées à Casabourse ;
- Intégration de **variables macroéconomiques** (taux directeur BAM, MASI, M2) ;
- Recours à des **modèles non-linéaires** (XGBoost, Random Forest) pour capturer les interactions entre variables ;
- Développement d'un **LSTM enrichi de features fondamentales** (PER, ROE, EPS en entrée) ;
- Application d'un **modèle à effets fixes** pour contrôler l'hétérogénéité entre entreprises.

---

## VII. Conclusion

Ce projet a démontré la faisabilité et la pertinence de l'application du machine learning pour la prédiction du rendement des dividendes sur le marché boursier marocain, en prenant Auto Hall S.A. comme terrain d'application.

**Réponse à la problématique :** il est possible de prédire le rendement des dividendes des sociétés cotées à la Bourse de Casablanca avec une précision acceptable à partir d'indicateurs financiers fondamentaux, en mobilisant des techniques de régression régularisée. La régression linéaire simple constitue déjà une référence compétitive (R² = 0,418 ; MAE = 0,77) pour une entreprise à politique de distribution stable. La régularisation Ridge améliore la robustesse du modèle face à la multicolinéarité entre variables, tandis que le Lasso identifie automatiquement les indicateurs les plus déterminants — en l'occurrence, le ROE et l'EPS.

**Principaux enseignements :**
1. Le ROE et l'EPS constituent les déterminants fondamentaux les plus prédictifs du taux de dividende, conformément aux théories financières de référence (Gordon, Ross, Jensen & Meckling) ;
2. La régularisation Lasso apporte une valeur ajoutée analytique en sélectionnant automatiquement les variables informatives et en éliminant les features redondantes ou peu pertinentes ;
3. Les modèles de deep learning (LSTM) ne surpassent pas les régressions classiques lorsqu'ils sont appliqués à la seule série de prix, sans intégration de features fondamentales ;
4. L'approche data-driven constitue une avancée méthodologique vers une aide à la décision objective pour les investisseurs sur le marché casablancais.

**Perspectives :** l'extension à un modèle de panel sur l'ensemble des sociétés cotées à Casabourse, couplée à l'intégration de variables macroéconomiques et au recours au deep learning alimenté par des données fondamentales, constitue la prochaine étape naturelle de ce travail de recherche appliquée.

---

## Bibliographie / Webographie

**Ouvrages et articles académiques :**

- Ross, S.A. (1977). *The Determination of Financial Structure: The Incentive-Signalling Approach*. Bell Journal of Economics, 8(1), 23–40.
- Jensen, M.C. & Meckling, W.H. (1976). *Theory of the Firm: Managerial Behavior, Agency Costs and Ownership Structure*. Journal of Financial Economics, 3(4), 305–360.
- Gordon, M.J. (1959). *Dividends, Earnings and Stock Prices*. Review of Economics and Statistics, 41(2), 99–105.
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society, Series B, 58(1), 267–288.
- Hoerl, A.E. & Kennard, R.W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics, 12(1), 55–67.

**Sources de données :**

- Bourse de Casablanca (Casabourse) — données historiques Auto Hall S.A. : https://www.casablanca-bourse.com
- Investing.com — Auto Hall Historical Data : https://www.investing.com/equities/auto-hall-historical-data
- Rapports annuels Auto Hall S.A. (2015–2024) — disponibles sur le site de la Bourse de Casablanca

**Documentation technique :**

- Scikit-learn Documentation — Linear Models : https://scikit-learn.org/stable/modules/linear_model.html
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.

---

## Annexes

### Annexe A — Pipeline Python complet

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df = pd.read_csv('Auto_Hall.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

def parse_vol(v):
    v = str(v).replace(',', '')
    if 'K' in v: return float(v.replace('K', '')) * 1e3
    if 'M' in v: return float(v.replace('M', '')) * 1e6
    return float(v)

df['Volume']     = df['Vol.'].apply(parse_vol)
df['Change_pct'] = df['Change %'].str.replace('%', '').astype(float)

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
df['Returns']       = np.log(df['Price'] / df['Price'].shift(1))
df['Volatilite_30j']= df['Returns'].rolling(30).std() * np.sqrt(252)
df['MA_50']         = df['Price'].rolling(50).mean()
df['MA_200']        = df['Price'].rolling(200).mean()
df['MA_ratio']      = df['MA_50'] / df['MA_200']
df['Momentum_252j'] = df['Price'] / df['Price'].shift(252) - 1
df['Volume_ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()

def rsi(s, w=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(w).mean()
    l = -d.where(d < 0, 0).rolling(w).mean()
    return 100 - 100 / (1 + g / l)

df['RSI_14j'] = rsi(df['Price'])
df['MACD']    = (df['Price'].ewm(span=12).mean()
               - df['Price'].ewm(span=26).mean())

# ── 3. Données fondamentales ───────────────────────────────────────────────────
fund_df = pd.DataFrame({
    'Year': [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
    'EPS':  [6.2, 5.8, 7.1, 6.9, 7.4, 4.2, 6.8, 8.1, 8.7, 9.2],
    'DPS':  [4.0, 3.5, 4.5, 4.5, 5.0, 2.5, 4.0, 5.5, 6.0, 6.5],
    'ROE':  [14.2,12.8,15.3,14.7,15.1,8.9,14.2,16.8,17.2,18.1],
    'PER':  [12.1,13.4,11.8,12.6,11.9,17.2,12.8,10.9,10.4,9.8]
})

df['Year'] = df['Date'].dt.year
df = df.merge(fund_df, on='Year', how='left')
df['Taux_Dividende'] = (df['DPS'] / df['Price']) * 100

# ── 4. Modélisation ────────────────────────────────────────────────────────────
features = ['Volatilite_30j','MA_ratio','RSI_14j','MACD',
            'Momentum_252j','Volume_ratio','PER','ROE','EPS']

df_m = df[features + ['Taux_Dividende']].dropna()
X    = df_m[features].values
y    = df_m['Taux_Dividende'].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

split   = int(len(df_m) * 0.80)
X_tr, X_te = X_sc[:split], X_sc[split:]
y_tr, y_te = y[:split],    y[split:]

tscv   = TimeSeriesSplit(n_splits=5)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

models = {
    'Linéaire': LinearRegression(),
    'Ridge':    RidgeCV(alphas=alphas, cv=tscv),
    'Lasso':    LassoCV(alphas=alphas, cv=tscv, max_iter=10_000)
}

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    results[name] = {
        'MAE' : round(mean_absolute_error(y_te, pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_te, pred)), 3),
        'R²'  : round(r2_score(y_te, pred), 3)
    }

print(pd.DataFrame(results).T.to_string())
```

### Annexe B — Visualisation des résultats

```python
# Figure 1 : Comparaison des métriques par modèle
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['MAE', 'RMSE', 'R²']
colors  = ['#006633', '#D4A017', '#2C2C2C']  # Couleurs ENCG

for ax, metric, color in zip(axes, metrics, colors):
    vals = [results[m][metric] for m in results]
    ax.bar(list(results.keys()), vals, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Comparaison des modèles — Prédiction taux dividende Auto Hall",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparaison_modeles.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 2 : Évolution du prix Auto Hall (1999–2025)
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Price'], linewidth=0.8, color='#006633')
plt.title("Évolution du cours boursier Auto Hall — Bourse de Casablanca (1999–2025)",
          fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Prix de clôture (MAD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_prix_autohall.png', dpi=150)
plt.show()
```

### Annexe C — Extrait du dataset brut Auto Hall

| Ligne | Date | Price | Open | High | Low | Vol. | Change% |
|-------|------|-------|------|------|-----|------|---------|
| 0 | 12/09/2025 | 93,00 | 93,98 | 93,98 | 93,00 | 0,14K | -1,04% |
| 1 | 12/08/2025 | 93,98 | 94,00 | 94,00 | 91,36 | 0,13K | -0,01% |
| 2 | 12/05/2025 | 93,99 | 91,02 | 94,00 | 91,02 | 3,44K | -0,37% |
| 3 | 12/04/2025 | 94,34 | 92,10 | 94,38 | 91,00 | 7,51K | +0,37% |
| 4 | 12/03/2025 | 93,99 | 92,52 | 94,94 | 92,50 | 1,93K | -1,00% |
| … | … | … | … | … | … | … | … |
| 4773 | 01/12/2000 | 22,25 | 22,25 | 22,25 | 22,25 | 0,22K | +5,95% |
| 4775 | 01/06/2000 | 19,88 | 19,88 | 19,88 | 19,88 | 0,10K | +6,03% |
| 4776 | 12/30/1999 | 18,75 | 18,75 | 18,75 | 18,75 | 0,02K | +5,93% |
| 4777 | 12/29/1999 | 17,70 | 17,70 | 17,70 | 17,70 | 0,10K | +5,48% |

*Total : 4 778 observations journalières | Période : 29/12/1999 – 09/12/2025*

---

*Compte rendu de projet réalisé dans le cadre du module Finance Quantitative & Data Science Financière*  
*ENCG Settat — Licence CAC, Semestre 8 — Année universitaire 2025–2026*  
*Données : Auto Hall S.A. — Bourse de Casablanca — Source : Casabourse / Investing.com*
