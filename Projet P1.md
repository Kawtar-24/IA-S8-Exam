# Rapport Détaillé — Algorithme de Clustering K-Means et Variantes

> **Type d'apprentissage :** Non Supervisé (Clustering)  
> **Bibliothèque principale :** `scikit-learn`  
> **Date :** Mars 2026

---

## Table des matières

1. [Introduction au Clustering et K-Means](#1-introduction-au-clustering-et-k-means)
2. [Préparation des données](#2-préparation-des-données)
3. [Les Algorithmes de la famille K-Means](#3-les-algorithmes-de-la-famille-k-means)
    - [3.1 K-Means Standard](#31-k-means-standard)
    - [3.2 Mini-Batch K-Means](#32-mini-batch-k-means)
    - [3.3 Bisecting K-Means](#33-bisecting-k-means)
4. [Comment déterminer le nombre optimal de clusters ($k$) ?](#4-comment-déterminer-le-nombre-optimal-de-clusters-k)
    - [La Méthode du Coude (Elbow Method)](#la-méthode-du-coude-elbow-method)
    - [Le Score de Silhouette](#le-score-de-silhouette)
5. [Avantages, Inconvénients et Cas d'usage](#5-avantages-inconvénients-et-cas-dusage)

---

## 1. Introduction au Clustering et K-Means

Le **clustering** (ou partitionnement de données) est une tâche d'apprentissage **non supervisé**. Contrairement à la classification où l'on cherche à prédire une étiquette (Y) à partir de caractéristiques (X), le clustering consiste à regrouper des données (X) non étiquetées en groupes homogènes, appelés *clusters*.

**Le K-Means** est l'algorithme de clustering le plus populaire. 
- **Objectif :** Diviser $N$ observations en $k$ clusters, de sorte que chaque observation appartienne au cluster dont la moyenne (le centroïde) est la plus proche.
- **Principe mathématique :** Il cherche à minimiser la **variance intra-cluster** (ou l'inertie), c'est-à-dire la somme des distances euclidiennes au carré entre chaque point et le centroïde de son cluster d'affectation.
- **Algorithme de Lloyd :** 
  1. Initialiser $k$ centroïdes aléatoirement (ou intelligemment avec `k-means++`).
  2. Assigner chaque point au centroïde le plus proche.
  3. Recalculer la position des centroïdes (moyenne des points du cluster).
  4. Répéter 2 et 3 jusqu'à convergence (les centroïdes ne bougent plus).

---

## 2. Préparation des données

Le K-Means repose sur la mesure de la distance (distance euclidienne). Il est donc **primordial** de mettre à l'échelle (standardiser) les données avant d'appliquer l'algorithme, sinon les variables avec de plus grandes amplitudes domineront le calcul de la distance.

```python
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Chargement de données d'exemple
X_iris, _ = load_iris(return_X_y=True)
X_digits, _ = load_digits(return_X_y=True)

# Bonnes pratiques : utiliser un Pipeline pour garantir la standardisation
# On n'a pas besoin de y (étiquettes) pour l'entraînement
```

---

## 3. Les Algorithmes de la famille K-Means

### 3.1 K-Means Standard

L'implémentation de base. Rapide pour des jeux de données de taille moyenne, mais sa complexité augmente avec le nombre d'échantillons et de clusters.

```python
from sklearn.cluster import KMeans

# 1. Définition du modèle avec standardisation
kmeans_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,          # Le nombre de groupes à former (k)
        init="k-means++",      # Stratégie d'initialisation pour accélérer la convergence
        n_init=10,             # L'algorithme sera lancé 10 fois, le meilleur résultat est gardé
        max_iter=300,          # Nombre maximum d'itérations pour une même exécution
        random_state=42        # Pour la reproductibilité
    ))
])

# 2. Apprentissage (sans y !)
kmeans_pipeline.fit(X_iris)

# 3. Récupération des résultats
modele_km = kmeans_pipeline.named_steps["kmeans"]
clusters_assignes = modele_km.labels_         # A quel cluster appartient chaque point (0, 1, ou 2)
centroides = modele_km.cluster_centers_       # Coordonnées des 3 centroïdes
inertie = modele_km.inertia_                  # Métrique d'évaluation interne (plus c'est bas, mieux c'est)

print(f"Inertie du modèle : {inertie:.2f}")
```

### 3.2 Mini-Batch K-Means

Une variante optimisée pour le **Big Data**. Au lieu d'utiliser l'ensemble du dataset à chaque itération pour mettre à jour les centroïdes, l'algorithme utilise un petit sous-échantillon aléatoire (mini-batch).

- **Avantage :** Extrêmement rapide, occupation mémoire très faible.
- **Inconvénient :** Qualité du clustering légèrement inférieure au K-Means standard (inertie un peu plus élevée).

```python
from sklearn.cluster import MiniBatchKMeans

mbk_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mbk", MiniBatchKMeans(
        n_clusters=10,
        batch_size=1024,       # Taille du lot traité à chaque itération
        max_iter=100,
        n_init=3,              # Moins d'initialisations nécessaires car très rapide
        random_state=42
    ))
])

mbk_pipeline.fit(X_digits)
# Note : on peut aussi utiliser mbk_pipeline.named_steps["mbk"].partial_fit(chunks) 
# pour des flux de données continus.
```

### 3.3 Bisecting K-Means

Une approche hiérarchique au lieu d'une approche "à plat". 
1. Tous les points commencent dans un seul grand cluster.
2. Le cluster ayant la plus grande inertie est divisé en 2 (avec un K-Means où $k=2$).
3. Le processus est répété jusqu'à obtenir les $k$ clusters demandés.

- **Avantage :** Produit souvent des clusters de tailles plus régulières que le K-Means classique.

```python
from sklearn.cluster import BisectingKMeans

bk_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("bkp", BisectingKMeans(
        n_clusters=10,
        init="k-means++",
        bisecting_strategy="biggest_inertia", # Split le cluster le plus dispersé
        random_state=42
    ))
])

bk_pipeline.fit(X_digits)
```

---

## 4. Comment déterminer le nombre optimal de clusters ($k$) ?

Puisque nous n'avons pas la vérité terrain, nous devons utiliser des heuristiques pour choisir le meilleur $k$.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

X_scaled = StandardScaler().fit_transform(X_iris)

inerties = []
silhouettes = []
k_valeurs = range(2, 10)

for k in k_valeurs:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
```

### La Méthode du Coude (Elbow Method)

On trace l'inertie en fonction de $k$. L'inertie baisse toujours quand $k$ augmente. On cherche le point "coude" où la baisse ralentit drastiquement.

```python
plt.figure(figsize=(8, 4))
plt.plot(k_valeurs, inerties, marker='o', linestyle='--')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie (Variance intra-cluster)")
plt.title("Méthode du Coude")
plt.grid(True)
plt.show()
# Interprétation : Si un coude net se forme à k=3, alors 3 est un bon choix.
```

### Le Score de Silhouette

La silhouette mesure la cohésion (proximité au sein du cluster) de chaque point vis-à-vis de la séparation (distance par rapport au cluster voisin le plus proche). 
Le score global varie de **-1 à 1** (plus proche de 1 = excellent clustering). On cherche à **maximiser** ce score.

```python
plt.figure(figsize=(8, 4))
plt.plot(k_valeurs, silhouettes, marker='s', color='orange')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de Silhouette Global")
plt.title("Score de Silhouette par cluster")
plt.grid(True)
plt.show()
# Interprétation : Le k donnant le score le plus élevé est souvent le plus robuste.
```

---

## 5. Avantages, Inconvénients et Cas d'usage

### Avantages
- **Simple et facile à comprendre.**
- **Rapide et scalable** (surtout Mini-Batch K-Means, complexité $O(N)$).
- **Implémentation scikit-learn très mature**, gère bien les données creuses (Sparse matrices).

### Inconvénients
- **Il faut choisir $k$ manuellement.**
- **Sensible aux outliers (valeurs aberrantes),** car ils tirent fortement sur la moyenne (le centroïde).
- **Sensible à l'échelle des données** (nécessite la standardisation).
- **Hypothèse de base forte :** Il suppose que les clusters sont sphériques et de variance similaire. Il échoue totalement sur des clusters de formes complexes (en demi-lune, orbites concentriques).

### Guide de choix

| Objectif / Contrainte | Recommandation |
|----------------------|----------------|
| Dataset de taille moyenne (< 100k) | **K-Means Standard** |
| Dataset massif / Big Data | **Mini-Batch K-Means** (gains majeurs de RAM et CPU) |
| Clusters de taille équilibrée souhaités | **Bisecting K-Means** |
| Détecter des anomalies / Outliers | (Utiliser un algorithme comme **DBSCAN** ou Isolation Forest plutôt que K-Means) |
| Détecter des clusters de formes non-sphériques | (Utiliser **DBSCAN**, **Spectral Clustering** ou HDBSCAN) |

---
*Généré automatiquement suite à l'import de votre rapport sur les algorithmes supervisés.*
