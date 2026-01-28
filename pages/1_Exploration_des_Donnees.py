import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("📊 Exploration des Données")
st.markdown("---")

if 'df' not in st.session_state:
    st.error("Les données n'ont pas été chargées. Veuillez retourner à la page d'accueil.")
    st.stop()

df = st.session_state['df']

st.header("Stroke Prediction Dataset")

st.markdown("""
### Contexte et Problématique

Les accidents vasculaires cérébraux (AVC) représentent une cause majeure de décès et d'invalidité dans le monde.
La prédiction précoce du risque d'AVC peut permettre une intervention médicale rapide et sauver des vies.

Ce dataset contient des informations sur des patients, incluant leurs caractéristiques démographiques,
leurs antécédents médicaux et leur statut concernant l'AVC. L'objectif est d'identifier les facteurs
de risque et de développer des modèles de prédiction fiables.

Cependant, il est crucial de s'assurer que ces modèles ne présentent pas de biais discriminatoires
envers certains groupes démographiques, notamment en fonction du genre ou de la zone géographique
(rurale vs urbaine). Une prédiction biaisée pourrait entraîner des disparités dans l'accès aux soins
et aux traitements préventifs.
""")

st.markdown("---")

st.subheader("📈 Métriques Clés (KPIs)")

col1, col2, col3, col4 = st.columns(4)

total_rows = len(df)
col1.metric("Nombre total de lignes", f"{total_rows:,}")

total_cols = len(df.columns)
col2.metric("Nombre de colonnes", total_cols)

missing_rate = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
col3.metric("Taux de valeurs manquantes", f"{missing_rate:.2f}%")

if 'stroke' in df.columns:
    stroke_rate = (df['stroke'].sum() / total_rows) * 100
    col4.metric("Taux d'AVC", f"{stroke_rate:.2f}%")
else:
    col4.metric("Variable cible", "Non trouvée")

st.markdown("---")

st.subheader("👀 Aperçu des Données")
st.dataframe(df.head(100), use_container_width=True)

st.subheader("📝 Description des Colonnes")

col_info = []
for col in df.columns:
    col_info.append({
        'Colonne': col,
        'Type': str(df[col].dtype),
        'Valeurs manquantes': df[col].isnull().sum(),
        'Valeurs uniques': df[col].nunique(),
        'Exemples': ', '.join([str(x) for x in df[col].dropna().unique()[:5]])
    })

col_df = pd.DataFrame(col_info)
st.dataframe(col_df, use_container_width=True)

col_descriptions = {
    'id': 'Identifiant unique du patient',
    'gender': 'Genre du patient (Male, Female, Other)',
    'age': 'Âge du patient (en années)',
    'hypertension': 'Présence d\'hypertension (0 = Non, 1 = Oui)',
    'heart_disease': 'Présence de maladie cardiaque (0 = Non, 1 = Oui)',
    'ever_married': 'Statut marital (Yes, No)',
    'work_type': 'Type de travail (Private, Self-employed, Govt_job, children, Never_worked)',
    'Residence_type': 'Type de résidence (Urban, Rural)',
    'avg_glucose_level': 'Niveau moyen de glucose dans le sang (mg/dL)',
    'bmi': 'Indice de masse corporelle (Body Mass Index)',
    'smoking_status': 'Statut tabagique (formerly smoked, never smoked, smokes, Unknown)',
    'stroke': 'Variable cible - Présence d\'AVC (0 = Non, 1 = Oui)'
}

st.markdown("#### Signification des Colonnes")
for col in df.columns:
    if col in col_descriptions:
        st.markdown(f"- **{col}** : {col_descriptions[col]}")

st.markdown("---")

st.subheader("📊 Visualisations")
if 'stroke' in df.columns:
    st.markdown("#### 1. Distribution de la variable cible (Stroke)")
    fig = px.histogram(
        df,
        x="stroke",
        title="Distribution de la variable cible (AVC)",
        labels={'stroke': 'AVC (0=Non, 1=Oui)', 'count': 'Nombre de patients'},
        color_discrete_sequence=['#FF6B6B']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patients sans AVC", f"{(df['stroke'] == 0).sum():,}")
    with col2:
        st.metric("Patients avec AVC", f"{(df['stroke'] == 1).sum():,}")

st.markdown("---")

if 'gender' in df.columns and 'stroke' in df.columns:
    st.markdown("#### 2. Comparaison par Genre")

    gender_stroke = df.groupby(['gender', 'stroke']).size().reset_index(name='count')
    fig = px.bar(
        gender_stroke,
        x='gender',
        y='count',
        color='stroke',
        title="Distribution des AVC par Genre",
        labels={'gender': 'Genre', 'count': 'Nombre de patients', 'stroke': 'AVC'},
        color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
    )
    st.plotly_chart(fig, use_container_width=True)

    gender_rates = df.groupby('gender')['stroke'].agg(['mean', 'count']).reset_index()
    gender_rates['taux_avc'] = gender_rates['mean'] * 100
    st.dataframe(gender_rates[['gender', 'taux_avc', 'count']].rename(columns={
        'gender': 'Genre',
        'taux_avc': 'Taux d\'AVC (%)',
        'count': 'Nombre de patients'
    }), use_container_width=True)

st.markdown("---")

if 'Residence_type' in df.columns and 'stroke' in df.columns:
    st.markdown("#### 3. Comparaison par Zone Géographique")

    residence_stroke = df.groupby(['Residence_type', 'stroke']).size().reset_index(name='count')
    fig = px.bar(
        residence_stroke,
        x='Residence_type',
        y='count',
        color='stroke',
        title="Distribution des AVC par Zone Géographique",
        labels={'Residence_type': 'Type de Résidence', 'count': 'Nombre de patients', 'stroke': 'AVC'},
        color_discrete_map={0: '#95E1D3', 1: '#F38181'}
    )
    st.plotly_chart(fig, use_container_width=True)

    residence_rates = df.groupby('Residence_type')['stroke'].agg(['mean', 'count']).reset_index()
    residence_rates['taux_avc'] = residence_rates['mean'] * 100
    st.dataframe(residence_rates[['Residence_type', 'taux_avc', 'count']].rename(columns={
        'Residence_type': 'Type de Résidence',
        'taux_avc': 'Taux d\'AVC (%)',
        'count': 'Nombre de patients'
    }), use_container_width=True)

st.markdown("---")

st.markdown("#### 4. Matrice de Corrélations")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        title="Matrice de Corrélations entre Variables Numériques",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if 'age' in df.columns:
    st.markdown("#### 5. Distribution de l'Âge par Statut AVC")
    fig = px.box(
        df,
        x='stroke',
        y='age',
        title="Distribution de l'Âge selon le Statut AVC",
        labels={'stroke': 'AVC (0=Non, 1=Oui)', 'age': 'Âge'},
        color='stroke',
        color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if 'gender' in df.columns:
    st.markdown("#### 6. Répartition par Genre")
    gender_counts = df['gender'].value_counts()
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Répartition des Patients par Genre"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔍 Filtres Interactifs")

with st.expander("Appliquer des filtres aux données"):
    col1, col2 = st.columns(2)

    filtered_df = df.copy()

    if 'gender' in df.columns:
        with col1:
            selected_genders = st.multiselect(
                "Sélectionner les genres",
                options=df['gender'].unique(),
                default=df['gender'].unique()
            )
            filtered_df = filtered_df[filtered_df['gender'].isin(selected_genders)]

    if 'Residence_type' in df.columns:
        with col2:
            selected_residences = st.multiselect(
                "Sélectionner les zones géographiques",
                options=df['Residence_type'].unique(),
                default=df['Residence_type'].unique()
            )
            filtered_df = filtered_df[filtered_df['Residence_type'].isin(selected_residences)]

    if 'age' in df.columns:
        age_range = st.slider(
            "Plage d'âge",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(int(df['age'].min()), int(df['age'].max()))
        )
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) &
            (filtered_df['age'] <= age_range[1])
        ]

    st.write(f"**Nombre de lignes après filtrage :** {len(filtered_df)}")
    st.dataframe(filtered_df.head(50), use_container_width=True)
