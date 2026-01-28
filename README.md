# 🧠 Application de Détection de Biais - Prédiction d'AVC

Application Streamlit pour l'exploration de données et la détection de biais dans le dataset Stroke Prediction.

## 📋 Description

Cette application permet d'analyser le dataset **Stroke Prediction Dataset** de Kaggle et de détecter les biais potentiels liés au **genre** et à la **zone géographique** (Rural/Urban) dans la prédiction du risque d'AVC.

## 📁 Structure du Projet

```
PROJET/
├── app.py                          # Point d'entrée principal de l'application
├── pages/                          # Dossier contenant les pages de l'application
│   ├── 1_Exploration_des_Donnees.py    # Page d'exploration des données
│   ├── 2_Detection_de_Biais.py         # Page de détection de biais
│   └── 3_Modelisation.py               # Page de modélisation (BONUS)
├── utils/                          # Dossier contenant les fonctions utilitaires
│   ├── __init__.py
│   └── fairness.py                 # Fonctions de calcul des métriques de fairness
├── requirements.txt                # Dépendances Python du projet
├── README.md                       # Documentation du projet
└── .gitignore                      # Fichiers à ignorer par Git
```

### Contenu des Fichiers

#### `app.py`
- **Fonction principale** : Point d'entrée de l'application Streamlit
- **Fonctionnalités** :
  - Configuration de la page (titre, icône, layout)
  - Téléchargement automatique du dataset depuis Kaggle via `kagglehub`
  - Nettoyage des données (normalisation de la colonne `age` en entier)
  - Mise en cache des données pour améliorer les performances
  - Stockage du dataframe dans `st.session_state` pour partage entre pages
  - Contenu de la page d'accueil (présentation du projet, contexte, problématique)

#### `pages/1_Exploration_des_Donnees.py`
- **Fonction principale** : Exploration et visualisation des données
- **Contenu** :
  - 4 métriques KPIs (nombre de lignes, colonnes, taux de valeurs manquantes, distribution de la variable cible)
  - Aperçu interactif du dataframe
  - Description technique et signification des colonnes
  - 6 visualisations interactives (histogrammes, barres, corrélations, box plots, pie charts)
  - Filtres interactifs pour explorer les données

#### `pages/2_Detection_de_Biais.py`
- **Fonction principale** : Détection et analyse des biais dans les données
- **Contenu** :
  - Explication des biais analysés (genre, zone géographique)
  - Calcul de 2 métriques de fairness (Parité Démographique, Ratio d'Impact Disproportionné)
  - Visualisations comparatives par groupe
  - Interprétation des résultats et recommandations

#### `pages/3_Modelisation.py`
- **Fonction principale** : Entraînement et évaluation de modèles de machine learning
- **Contenu** :
  - Sélection des features et attributs sensibles
  - Préparation des données (encodage, gestion des valeurs manquantes)
  - Entraînement de modèles (Logistic Regression, Random Forest)
  - Calcul des métriques de performance (Accuracy, Precision, Recall, F1-Score)
  - Calcul des métriques de fairness sur les prédictions
  - Comparaison des performances par groupe sensible
  - Matrices de confusion par groupe
  - Gestion du déséquilibre de classes avec `class_weight='balanced'`

#### `utils/fairness.py`
- **Fonction principale** : Implémentation des métriques de fairness
- **Fonctions** :
  - `demographic_parity_difference()` : Calcule la différence de parité démographique
  - `disparate_impact_ratio()` : Calcule le ratio d'impact disproportionné
  - `equalized_odds_difference()` : Calcule la différence d'égalité des chances

## 🚀 Installation

1. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

2. **Configurer Kaggle** (pour télécharger le dataset) :
   - Créer un compte sur [Kaggle](https://www.kaggle.com/)
   - Télécharger votre fichier `kaggle.json` depuis les paramètres de votre compte
   - Placer le fichier dans `~/.kaggle/` (Linux/Mac) ou `C:\Users\<username>\.kaggle\` (Windows)

## ▶️ Lancement

Lancer l'application avec :
```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📑 Structure de l'Application

### Page 1 : 🏠 Accueil
- **Titre et présentation du dataset** : Description du Stroke Prediction Dataset
- **Contexte et problématique** : Explication de l'importance de la prédiction d'AVC et des enjeux liés aux biais
- **Navigation** : Accès aux autres pages via la barre latérale de Streamlit

### Page 2 : 📊 Exploration des Données
- **4 KPIs** : 
  - Nombre total de lignes
  - Nombre de colonnes
  - Taux de valeurs manquantes
  - Distribution de la variable cible (taux d'AVC)
- **Aperçu des données** : DataFrame interactif avec les 100 premières lignes
- **Description des colonnes** : Tableau avec type, valeurs manquantes, valeurs uniques et exemples
- **Signification des colonnes** : Explications en langage naturel de chaque colonne
- **6 visualisations** :
  1. Distribution de la variable cible (histogramme)
  2. Comparaison par genre (graphique en barres)
  3. Comparaison par zone géographique (graphique en barres)
  4. Matrice de corrélations (heatmap)
  5. Distribution de l'âge par statut AVC (box plot)
  6. Répartition par genre (pie chart)
- **Filtres interactifs** (BONUS) : Filtrage par genre, zone géographique, statut marital, etc.

### Page 3 : ⚠️ Détection de Biais
- **Sélection de l'attribut sensible** : Choix entre genre ou zone géographique
- **Explication du biais analysé** : Pourquoi c'est problématique et quel est l'impact réel
- **2 métriques de fairness** :
  1. **Parité Démographique** : Différence maximale entre les taux de prédiction positive par groupe
  2. **Ratio d'Impact Disproportionné** : Ratio entre le taux du groupe non-privilégié et celui du groupe privilégié
- **Visualisations des résultats** : Graphiques en barres comparant les taux par groupe
- **Interprétation** : Analyse concrète du biais détecté, identification du groupe défavorisé, impact réel et recommandations

### Page 4 : 🤖 Modélisation (BONUS)
- **Préparation des données** :
  - Sélection des variables à utiliser comme features
  - Choix de l'attribut sensible pour l'analyse de fairness
  - Gestion des valeurs manquantes (médiane pour numériques, mode pour catégorielles)
  - Encodage des variables catégorielles
- **Sélection du modèle** :
  - Logistic Regression ou Random Forest
  - Option `class_weight='balanced'` pour gérer le déséquilibre de classes
  - Paramètres ajustables pour Random Forest (nombre d'arbres, profondeur max)
- **Performances globales** : Accuracy, Precision, Recall, F1-Score
- **Distribution des prédictions** : Visualisation du nombre de prédictions par classe
- **Métriques de fairness sur les prédictions** : Parité Démographique et Ratio d'Impact Disproportionné
- **Performances par groupe** : Comparaison des métriques (Accuracy, Precision, Recall, F1-Score) pour chaque groupe sensible
- **Matrices de confusion par groupe** : Visualisation des vrais/faux positifs et négatifs pour chaque groupe

## 📚 Définitions des Termes Techniques

### Métriques de Performance

#### **Accuracy (Précision Globale)**
Pourcentage de prédictions correctes parmi toutes les prédictions. Formule : `(Vrais Positifs + Vrais Négatifs) / Total`

**Exemple** : Si un modèle prédit correctement 95% des cas, l'Accuracy est de 0.95.

**Limitation** : Peut être trompeuse en cas de déséquilibre de classes. Un modèle qui prédit toujours la classe majoritaire aura une Accuracy élevée mais sera inutile.

#### **Precision (Précision)**
Proportion de prédictions positives qui sont réellement positives. Formule : `Vrais Positifs / (Vrais Positifs + Faux Positifs)`

**Exemple** : Si le modèle prédit 100 AVC et que 80 sont réellement des AVC, la Precision est de 0.80.

**Interprétation** : Mesure la fiabilité des prédictions positives. Une Precision élevée signifie peu de faux positifs.

#### **Recall (Rappel ou Sensibilité)**
Proportion de cas positifs réels qui sont correctement identifiés. Formule : `Vrais Positifs / (Vrais Positifs + Faux Négatifs)`

**Exemple** : S'il y a 100 AVC réels et que le modèle en détecte 78, le Recall est de 0.78.

**Interprétation** : Mesure la capacité du modèle à trouver tous les cas positifs. Un Recall élevé signifie peu de faux négatifs (cas manqués).

#### **F1-Score**
Moyenne harmonique entre Precision et Recall. Formule : `2 × (Precision × Recall) / (Precision + Recall)`

**Exemple** : Si Precision = 0.80 et Recall = 0.75, alors F1-Score = 2 × (0.80 × 0.75) / (0.80 + 0.75) ≈ 0.77

**Interprétation** : Équilibre entre Precision et Recall. Utile quand il faut trouver un compromis entre éviter les faux positifs et les faux négatifs.

### Métriques de Fairness (Équité)

#### **Parité Démographique (Demographic Parity)**
Mesure la différence maximale entre les taux de prédiction positive par groupe démographique. Plus la valeur est proche de 0, plus le modèle est équitable.

**Formule** : `max(taux_groupe_i) - min(taux_groupe_j)` pour tous les groupes i, j

**Exemple** : Si le groupe A a un taux de prédiction positive de 0.15 et le groupe B de 0.10, la différence est de 0.05.

**Interprétation** : 
- **0.0** : Parfaite équité (tous les groupes ont le même taux de prédiction positive)
- **> 0.1** : Biais potentiel significatif

#### **Ratio d'Impact Disproportionné (Disparate Impact Ratio)**
Ratio entre le taux de prédiction positive du groupe avec le taux le plus faible et celui du groupe avec le taux le plus élevé (groupe de référence). Mesure si un groupe est systématiquement désavantagé.

**Formule** : `taux_groupe_taux_faible / taux_groupe_taux_élevé`

**Note** : Dans cette application, les groupes "privilégié" et "non-privilégié" sont déterminés dynamiquement en fonction des taux réels observés dans les données, sans présupposer qu'un groupe spécifique est privilégié.

**Exemple** : Si le groupe A a un taux de 0.10 et le groupe B (référence) de 0.15, le ratio est 0.10/0.15 = 0.67.

**Interprétation** :
- **Entre 0.8 et 1.25** : Acceptable (pas de biais significatif)
- **< 0.8** : Le groupe non-privilégié est désavantagé (trop peu de prédictions positives)
- **> 1.25** : Le groupe privilégié est désavantagé (trop de prédictions positives)

#### **Égalité des Chances (Equalized Odds)**
Mesure si le modèle a les mêmes taux de vrais positifs (TPR) et de faux positifs (FPR) pour tous les groupes. Plus les différences sont proches de 0, plus le modèle est équitable.

**Formule** : 
- `TPR_diff = max(TPR_groupe_i) - min(TPR_groupe_j)`
- `FPR_diff = max(FPR_groupe_i) - min(FPR_groupe_j)`

**Interprétation** : Un modèle équitable devrait avoir les mêmes performances (taux d'erreurs) pour tous les groupes.

### Termes Généraux

#### **KPIs (Key Performance Indicators)**
Indicateurs clés de performance. Métriques importantes qui donnent une vue d'ensemble rapide de l'état des données ou du modèle.

#### **Variable Cible (Target Variable)**
Variable que l'on cherche à prédire. Dans ce projet, c'est la colonne `stroke` (0 = pas d'AVC, 1 = AVC).

#### **Features (Caractéristiques)**
Variables d'entrée utilisées pour faire des prédictions. Exemples : âge, genre, hypertension, etc.

#### **Attribut Sensible (Sensitive Attribute)**
Caractéristique démographique qui pourrait être source de discrimination. Dans ce projet : genre et zone géographique.

#### **Groupe Privilégié / Non-Privilégié**
Dans cette application, ces termes sont utilisés de manière technique pour le calcul des métriques de fairness :
- **Groupe privilégié (référence)** : Groupe avec le taux de prédiction positive le plus élevé dans les données observées
- **Groupe non-privilégié (comparé)** : Groupe avec le taux de prédiction positive le plus faible dans les données observées

**Note importante** : La détermination de ces groupes se fait automatiquement et dynamiquement en fonction des données réelles, sans présupposer qu'un groupe démographique spécifique est historiquement privilégié. L'objectif est de détecter les disparités dans les taux de prédiction, quelle que soit leur direction.

#### **Déséquilibre de Classes (Class Imbalance)**
Situation où une classe (ex: pas d'AVC) est beaucoup plus fréquente que l'autre (ex: AVC). Cela peut amener le modèle à toujours prédire la classe majoritaire.

**Solution** : Utiliser `class_weight='balanced'` pour donner plus de poids aux exemples de la classe minoritaire.

#### **Matrice de Confusion (Confusion Matrix)**
Tableau qui montre les prédictions correctes et incorrectes :
- **Vrais Positifs (TP)** : Cas positifs correctement prédits
- **Vrais Négatifs (TN)** : Cas négatifs correctement prédits
- **Faux Positifs (FP)** : Cas négatifs incorrectement prédits comme positifs
- **Faux Négatifs (FN)** : Cas positifs incorrectement prédits comme négatifs

#### **Encodage (Encoding)**
Conversion de variables catégorielles (texte) en nombres pour que les modèles puissent les utiliser. Exemple : "Male" → 0, "Female" → 1.

#### **Train/Test Split**
Division des données en deux ensembles :
- **Train** : Utilisé pour entraîner le modèle
- **Test** : Utilisé pour évaluer les performances du modèle sur des données jamais vues

#### **Logistic Regression**
Modèle de machine learning linéaire qui prédit la probabilité qu'un événement se produise. Adapté pour la classification binaire (0 ou 1).

#### **Random Forest**
Modèle de machine learning qui combine plusieurs arbres de décision pour faire des prédictions plus robustes. Moins sensible au surapprentissage que les arbres individuels.

## 📊 Métriques de Fairness - Détails

### Parité Démographique
Mesure la différence maximale entre les taux de prédiction positive par groupe. Plus proche de 0 = plus équitable.

**Utilisation** : Détecte si certains groupes reçoivent systématiquement plus ou moins de prédictions positives que d'autres.

### Ratio d'Impact Disproportionné (DI)
Ratio entre le taux de prédiction positive du groupe non-privilégié et celui du groupe privilégié. 
- **Ratio entre 0.8 et 1.25** = acceptable
- **Ratio < 0.8 ou > 1.25** = biais potentiel

**Utilisation** : Standard légal utilisé aux États-Unis pour détecter la discrimination. Un ratio < 0.8 indique une discrimination potentielle.

## 🛠️ Technologies Utilisées

- **Streamlit** : Framework Python pour créer des applications web interactives rapidement
- **Pandas** : Bibliothèque Python pour la manipulation et l'analyse de données
- **NumPy** : Bibliothèque Python pour les calculs numériques
- **Plotly** : Bibliothèque Python pour créer des visualisations interactives
- **Scikit-learn** : Bibliothèque Python pour le machine learning (modèles, métriques, préprocessing)
- **Kagglehub** : Bibliothèque Python pour télécharger facilement des datasets depuis Kaggle

## 📝 Notes Importantes

- **Téléchargement automatique** : Le dataset est automatiquement téléchargé lors du premier lancement via `kagglehub`
- **Mise en cache** : Les données sont mises en cache avec `@st.cache_data` pour améliorer les performances lors des rechargements
- **Nettoyage des données** : La colonne `age` est automatiquement normalisée en entier (valeurs manquantes remplacées par la médiane)
- **Connexion internet requise** : L'application nécessite une connexion internet pour télécharger le dataset la première fois
- **Gestion du déséquilibre** : La page de modélisation propose l'option `class_weight='balanced'` pour gérer le déséquilibre de classes
- **Navigation** : L'application utilise la navigation native multi-pages de Streamlit (dossier `pages/`)

## 🔗 Liens Utiles

- [Dataset Kaggle - Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Documentation Plotly](https://plotly.com/python/)
- [Documentation Scikit-learn](https://scikit-learn.org/stable/)
- [Fairness in Machine Learning - Google](https://developers.google.com/machine-learning/fairness-overview)
