import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.title("🤖 Modélisation")
st.markdown("---")

if 'df' not in st.session_state:
    st.error("Les données n'ont pas été chargées. Veuillez retourner à la page d'accueil.")
    st.stop()

df = st.session_state['df'].copy()

if 'stroke' not in df.columns:
    st.error("La variable cible 'stroke' n'a pas été trouvée dans le dataset.")
    st.stop()

st.header("Entraînement et Évaluation de Modèles")

st.markdown("""
Cette page permet d'entraîner des modèles de prédiction d'AVC et d'évaluer leurs performances
globales ainsi que leurs performances par groupe démographique pour détecter d'éventuels biais.
""")

st.markdown("---")

st.subheader("🔧 Préparation des Données")

st.markdown("#### Sélection des Variables")

exclude_cols = ['id', 'stroke']
available_features = [col for col in df.columns if col not in exclude_cols]

selected_features = st.multiselect(
    "Sélectionner les variables à utiliser comme features",
    options=available_features,
    default=available_features[:min(5, len(available_features))]
)

if not selected_features:
    st.warning("Veuillez sélectionner au moins une variable.")
    st.stop()

sensitive_attr = st.selectbox(
    "Attribut sensible pour l'analyse de fairness",
    options=['gender', 'Residence_type'],
    format_func=lambda x: 'Genre' if x == 'gender' else 'Zone Géographique'
)

X = df[selected_features].copy()
y = df['stroke'].copy()
sensitive = df[sensitive_attr].copy()

# handle missing values and encode categorical features
if X.isnull().sum().sum() > 0:
    st.warning("Des valeurs manquantes détectées. Elles seront remplies avec la médiane (numériques) ou le mode (catégorielles).")
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col].fillna(X[col].median(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else '', inplace=True)

label_encoders = {}
X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

st.markdown("#### Aperçu des Données Préparées")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Nombre d'échantillons :** {len(X_encoded)}")
    st.write(f"**Nombre de features :** {len(selected_features)}")
with col2:
    st.write(f"**Taux d'AVC :** {y.mean()*100:.2f}%")
    st.write(f"**Attribut sensible :** {sensitive_attr}")

st.markdown("---")

st.subheader("🎯 Sélection du Modèle")

model_type = st.selectbox(
    "Choisir le type de modèle",
    options=['Logistic Regression', 'Random Forest'],
    index=0
)

use_class_weight = st.checkbox(
    "Utiliser class_weight='balanced' pour gérer le déséquilibre des classes",
    value=True,
    help="Cette option équilibre automatiquement les poids des classes pour éviter que le modèle prédise toujours la classe majoritaire"
)

if model_type == 'Random Forest':
    n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
    max_depth = st.slider("Profondeur maximale", 3, 20, 10)

test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100

if st.button("🚀 Entraîner le Modèle"):
    with st.spinner("Entraînement en cours..."):
        # train/test split with stratification
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X_encoded, y, sensitive, test_size=test_size, random_state=42, stratify=y
        )

        # scale features for logistic regression
        if model_type == 'Logistic Regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        else:
            X_train_final = X_train
            X_test_final = X_test

        # train model
        if model_type == 'Logistic Regression':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced' if use_class_weight else None
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight='balanced' if use_class_weight else None
            )

        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]

        st.session_state['model'] = model
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['sensitive_test'] = sensitive_test
        st.session_state['model_type'] = model_type
        st.session_state['scaler'] = scaler if model_type == 'Logistic Regression' else None

        st.success("✅ Modèle entraîné avec succès !")

if 'model' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Performances Globales")

    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1-Score", f"{f1:.4f}")

    # warning if model predicts only one class
    unique_predictions = np.unique(y_pred)
    if len(unique_predictions) == 1:
        predicted_class = unique_predictions[0]
        count_class_0 = (y_pred == 0).sum()
        count_class_1 = (y_pred == 1).sum()
        st.warning(f"⚠️ **Problème détecté** : Le modèle prédit toujours la classe {predicted_class}. "
                  f"Prédictions : {count_class_0} fois classe 0, {count_class_1} fois classe 1. "
                  f"Cela explique pourquoi Precision, Recall et F1-Score sont à 0. "
                  f"Essayez d'activer 'class_weight=balanced' ou d'ajuster les paramètres du modèle.")

    # distribution des prédictions
    st.markdown("#### Distribution des Prédictions")
    pred_dist = pd.DataFrame({
        'Classe': ['Pas d\'AVC (0)', 'AVC (1)'],
        'Nombre': [(y_pred == 0).sum(), (y_pred == 1).sum()]
    })
    fig = px.bar(
        pred_dist,
        x='Classe',
        y='Nombre',
        title="Distribution des Prédictions du Modèle",
        color='Classe',
        color_discrete_map={'Pas d\'AVC (0)': '#4ECDC4', 'AVC (1)': '#FF6B6B'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Matrice de Confusion Globale")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
        x=['Sans AVC', 'Avec AVC'],
        y=['Sans AVC', 'Avec AVC'],
        title="Matrice de Confusion",
        color_continuous_scale='Blues',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("⚖️ Analyse de Fairness")

    sensitive_test = st.session_state['sensitive_test']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parité Démographique")
        dp_result = demographic_parity_difference(
            y_true=y_test.values,
            y_pred=y_pred,
            sensitive_attribute=sensitive_test.values
        )
        st.metric("Différence de Parité", f"{dp_result['difference']:.4f}")

        st.markdown("**Taux de prédiction positive par groupe :**")
        for group, rate in dp_result['rates'].items():
            st.write(f"- {group}: {rate:.4f} ({rate*100:.2f}%)")

    with col2:
        st.markdown("#### Ratio d'Impact Disproportionné")

        unique_vals = sensitive_test.unique()
        if len(unique_vals) >= 2:
            # déterminer dynamiquement les groupes basé sur les taux de prédiction réels
            group_rates = {}
            for group in unique_vals:
                group_mask = sensitive_test == group
                if group_mask.sum() > 0:
                    group_rates[group] = y_pred[group_mask].mean()
            
            if len(group_rates) >= 2:
                # groupe avec le taux le plus élevé = groupe de référence (privilégié)
                # groupe avec le taux le plus faible = groupe comparé (non-privilégié)
                privileged = max(group_rates, key=group_rates.get)
                unprivileged = min(group_rates, key=group_rates.get)
                
                di_result = disparate_impact_ratio(
                    y_true=y_test.values,
                    y_pred=y_pred,
                    sensitive_attribute=sensitive_test.values,
                    unprivileged_value=unprivileged,
                    privileged_value=privileged
                )

                st.metric("Ratio DI", f"{di_result['ratio']:.4f}")
                st.markdown("**Taux par groupe :**")
                st.write(f"- {unprivileged} (taux le plus faible): {di_result['unprivileged_rate']:.4f} ({di_result['unprivileged_rate']*100:.2f}%)")
                st.write(f"- {privileged} (taux le plus élevé - référence): {di_result['privileged_rate']:.4f} ({di_result['privileged_rate']*100:.2f}%)")
            else:
                st.warning("Pas assez de données pour calculer le ratio DI")

    st.markdown("---")
    st.subheader("📈 Performances par Groupe")

    groups = sensitive_test.unique()
    group_metrics = []

    for group in groups:
        group_mask = sensitive_test == group
        y_test_group = y_test[group_mask]
        y_pred_group = y_pred[group_mask]

        if len(y_test_group) > 0:
            group_metrics.append({
                'Groupe': group,
                'Accuracy': accuracy_score(y_test_group, y_pred_group),
                'Precision': precision_score(y_test_group, y_pred_group, zero_division=0),
                'Recall': recall_score(y_test_group, y_pred_group, zero_division=0),
                'F1-Score': f1_score(y_test_group, y_pred_group, zero_division=0),
                'Taille': len(y_test_group)
            })

    metrics_df = pd.DataFrame(group_metrics)
    st.dataframe(metrics_df, use_container_width=True)

    fig = go.Figure()

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Groupe'],
            y=metrics_df[metric],
            text=[f'{v:.3f}' for v in metrics_df[metric]],
            textposition='outside'
        ))

    fig.update_layout(
        title="Performances par Groupe",
        xaxis_title="Groupe",
        yaxis_title="Score",
        barmode='group',
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Matrices de Confusion par Groupe")

    cols = st.columns(len(groups))
    for idx, group in enumerate(groups):
        with cols[idx]:
            group_mask = sensitive_test == group
            y_test_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]

            cm_group = confusion_matrix(y_test_group, y_pred_group)
            fig = px.imshow(
                cm_group,
                labels=dict(x="Prédit", y="Réel", color="Nombre"),
                x=['Sans AVC', 'Avec AVC'],
                y=['Sans AVC', 'Avec AVC'],
                title=f"Groupe: {group}",
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Interprétation")

    st.markdown("""
    **Analyse des résultats :**
    
    Les métriques de fairness et les performances par groupe permettent d'identifier d'éventuels biais
    dans les prédictions du modèle. Des différences significatives entre les groupes peuvent indiquer :
    
    - **Biais dans les données d'entraînement** : Certains groupes peuvent être sous-représentés
    - **Biais dans les features** : Les variables utilisées peuvent être corrélées avec l'attribut sensible
    - **Biais algorithmique** : Le modèle peut apprendre des patterns discriminatoires
    
    **Recommandations :**
    
    1. Examiner les différences de performance entre groupes
    2. Ajuster les seuils de décision par groupe si nécessaire
    3. Utiliser des techniques de debiasing (fairness constraints, adversarial training, etc.)
    4. Collecter plus de données pour les groupes sous-représentés
    """)
