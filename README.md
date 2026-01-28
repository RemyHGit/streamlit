# 🧠 Application de Détection de Biais - Prédiction d'AVC

## 📊 Description

Les accidents vasculaires cérébraux (AVC) représentent une cause majeure de mortalité et d'invalidité dans le monde. La prédiction précoce du risque d'AVC peut permettre une intervention médicale rapide et sauver des vies. Cependant, il est crucial de s'assurer que les modèles de prédiction ne présentent pas de biais discriminatoires envers certains groupes démographiques, notamment en fonction du genre ou de la zone géographique (rurale vs urbaine).

Cette application Streamlit permet d'analyser le **Stroke Prediction Dataset** de Kaggle et de détecter les biais potentiels dans la prédiction du risque d'AVC. L'objectif est double : explorer les données de manière approfondie et identifier d'éventuels biais liés au **genre** et à la **zone géographique** (Rural/Urban) qui pourraient entraîner des disparités dans l'accès aux soins et aux traitements préventifs.

L'application offre une interface interactive pour visualiser les données, calculer des métriques de fairness (équité), entraîner des modèles de machine learning et évaluer leurs performances par groupe démographique afin de garantir un traitement équitable pour tous les patients.

## 🎯 Parcours

**Parcours A : Détection de Biais**

## 📁 Dataset

**Source** : [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**Taille** : ~5 110 lignes, 12 colonnes

**Variables principales** :
- `id` : Identifiant unique du patient
- `gender` : Genre du patient (Male, Female, Other)
- `age` : Âge du patient (en années)
- `hypertension` : Présence d'hypertension (0 = Non, 1 = Oui)
- `heart_disease` : Présence de maladie cardiaque (0 = Non, 1 = Oui)
- `ever_married` : Statut marital (Yes, No)
- `work_type` : Type de travail (Private, Self-employed, Govt_job, children, Never_worked)
- `Residence_type` : Type de résidence (Urban, Rural)
- `avg_glucose_level` : Niveau moyen de glucose dans le sang (mg/dL)
- `bmi` : Indice de masse corporelle (Body Mass Index)
- `smoking_status` : Statut tabagique (formerly smoked, never smoked, smokes, Unknown)

**Variable cible** : `stroke` - Présence d'AVC (0 = Non, 1 = Oui)

## 🚀 Fonctionnalités

### Page 1 : 🏠 Accueil
- Titre et présentation du dataset Stroke Prediction
- Contexte et problématique (2-3 paragraphes)
- Explication de l'importance de la prédiction d'AVC
- Enjeux liés aux biais dans les modèles de prédiction
- Navigation vers les autres pages via la barre latérale Streamlit

### Page 2 : 📊 Exploration des Données
- **4 métriques KPIs** :
  - Nombre total de lignes
  - Nombre de colonnes
  - Taux de valeurs manquantes
  - Distribution de la variable cible (taux d'AVC)
- **Aperçu des données** : DataFrame interactif avec les 100 premières lignes
- **Description technique des colonnes** : Tableau avec type, valeurs manquantes, valeurs uniques et exemples
- **Signification des colonnes** : Explications en langage naturel de chaque colonne
- **6 visualisations interactives** :
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
  2. **Ratio d'Impact Disproportionné** : Ratio entre le taux du groupe avec le taux le plus faible et celui du groupe avec le taux le plus élevé (référence)
- **Visualisations des résultats** : Graphiques en barres comparant les taux par groupe
- **Comparaison détaillée** : Distribution des AVC par groupe avec graphiques interactifs
- **Interprétation** : Analyse concrète du biais détecté, identification des disparités, impact réel et recommandations pour réduire le biais
- **Résumé des métriques** : Tableau récapitulatif avec valeurs et interprétations

### Page 4 (Bonus) : 🤖 Modélisation
- **Préparation des données** :
  - Sélection des variables à utiliser comme features
  - Choix de l'attribut sensible pour l'analyse de fairness
  - Gestion automatique des valeurs manquantes (médiane pour numériques, mode pour catégorielles)
  - Encodage automatique des variables catégorielles
- **Sélection du modèle** :
  - Logistic Regression ou Random Forest
  - Option `class_weight='balanced'` pour gérer le déséquilibre de classes
  - Paramètres ajustables pour Random Forest (nombre d'arbres, profondeur max)
  - Réglage de la taille du jeu de test
- **Performances globales** : Accuracy, Precision, Recall, F1-Score avec explications
- **Distribution des prédictions** : Visualisation du nombre de prédictions par classe
- **Avertissements** : Détection automatique si le modèle prédit toujours une seule classe
- **Métriques de fairness sur les prédictions** :
  - Parité Démographique
  - Ratio d'Impact Disproportionné
- **Performances par groupe** : Comparaison des métriques (Accuracy, Precision, Recall, F1-Score) pour chaque groupe sensible dans un tableau interactif
- **Matrices de confusion par groupe** : Visualisation des vrais/faux positifs et négatifs pour chaque groupe avec heatmaps

## 🛠️ Technologies Utilisées

- **Python 3.x**
- **Streamlit** : Framework pour créer des applications web interactives rapidement
- **Pandas** : Manipulation et analyse de données
- **NumPy** : Calculs numériques
- **Plotly Express** : Visualisations interactives
- **Scikit-learn** : Machine learning (modèles, métriques, préprocessing)
- **Kagglehub** : Téléchargement facile des datasets depuis Kaggle

## 📦 Installation Locale

```bash
# Cloner le repository
git clone https://github.com/RemyHGit/streamlit.git
cd streamlit

# Installer les dépendances
pip install -r requirements.txt

# Configurer Kaggle (pour télécharger le dataset)
# 1. Créer un compte sur https://www.kaggle.com/
# 2. Télécharger votre fichier kaggle.json depuis les paramètres de votre compte
# 3. Placer le fichier dans ~/.kaggle/ (Linux/Mac) ou C:\Users\<username>\.kaggle\ (Windows)

# Lancer l'application
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 🌐 Déploiement

Application déployée sur Streamlit Cloud : 👉 [Lien vers l'application](https://votre-app.streamlit.app)

*(À compléter avec le lien de déploiement réel)*

## 👥 Équipe

*À compléter avec les noms des membres de l'équipe*

## 📝 Notes

### Fonctionnalités techniques
- **Téléchargement automatique** : Le dataset est automatiquement téléchargé lors du premier lancement via `kagglehub`
- **Mise en cache** : Les données sont mises en cache avec `@st.cache_data` pour améliorer les performances lors des rechargements
- **Nettoyage des données** : La colonne `age` est automatiquement normalisée en entier (valeurs manquantes remplacées par la médiane)
- **Connexion internet requise** : L'application nécessite une connexion internet pour télécharger le dataset la première fois
- **Gestion du déséquilibre** : La page de modélisation propose l'option `class_weight='balanced'` pour gérer le déséquilibre de classes
- **Navigation** : L'application utilise la navigation native multi-pages de Streamlit (dossier `pages/`)

### Définitions des métriques

**Métriques de Performance** :
- **Accuracy** : Pourcentage de prédictions correctes parmi toutes les prédictions
- **Precision** : Proportion de prédictions positives qui sont réellement positives
- **Recall** : Proportion de cas positifs réels qui sont correctement identifiés
- **F1-Score** : Moyenne harmonique entre Precision et Recall

**Métriques de Fairness** :
- **Parité Démographique** : Différence maximale entre les taux de prédiction positive par groupe (plus proche de 0 = plus équitable)
- **Ratio d'Impact Disproportionné** : Ratio entre le taux du groupe avec le taux le plus faible et celui du groupe avec le taux le plus élevé. Entre 0.8 et 1.25 = acceptable

*Note* : Les groupes "privilégié" et "non-privilégié" sont déterminés dynamiquement en fonction des taux réels observés dans les données, sans présupposer qu'un groupe démographique spécifique est historiquement privilégié.

### Améliorations futures
- Ajout d'autres métriques de fairness (Equalized Odds, etc.)
- Support de plus d'attributs sensibles
- Export des résultats d'analyse
- Comparaison de plusieurs modèles simultanément

## 🔗 Liens Utiles

- [Dataset Kaggle - Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Documentation Plotly](https://plotly.com/python/)
- [Documentation Scikit-learn](https://scikit-learn.org/stable/)
- [Fairness in Machine Learning - Google](https://developers.google.com/machine-learning/fairness-overview)
