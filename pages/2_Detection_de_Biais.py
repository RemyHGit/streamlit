import streamlit as st
import pandas as pd
import plotly.express as px
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.title("⚠️ Détection de Biais")
st.markdown("---")

if 'df' not in st.session_state:
    st.error("Les données n'ont pas été chargées. Veuillez retourner à la page d'accueil.")
    st.stop()

df = st.session_state['df']

if 'stroke' not in df.columns:
    st.error("La variable cible 'stroke' n'a pas été trouvée dans le dataset.")
    st.stop()

st.header("Analyse des Biais dans la Prédiction d'AVC")

st.subheader("🔍 Sélection de l'Attribut Sensible")

sensitive_attr = st.selectbox(
    "Choisir l'attribut sensible à analyser",
    options=['gender', 'Residence_type'],
    format_func=lambda x: 'Genre' if x == 'gender' else 'Zone Géographique (Rural/Urban)'
)

if sensitive_attr not in df.columns:
    st.error(f"L'attribut '{sensitive_attr}' n'existe pas dans le dataset.")
    st.stop()

st.markdown("---")

st.subheader("📖 Explication du Biais Analysé")

if sensitive_attr == 'gender':
    st.markdown("""
    ### Attribut Sensible : Genre
    
    **Pourquoi c'est problématique ?**
    
    Les différences de genre dans la détection et le traitement des AVC peuvent avoir des conséquences graves :
    - Différents genres peuvent présenter des symptômes d'AVC différents
    - Les modèles entraînés sur des données déséquilibrées peuvent sous-estimer ou sur-estimer le risque pour certains genres
    - Cela peut entraîner des retards dans le diagnostic et le traitement, augmentant la mortalité et les séquelles
    
    **Impact réel** : Un biais dans la prédiction pourrait signifier que certains groupes à risque élevé ne recevraient pas les soins préventifs appropriés, tandis que d'autres groupes pourraient être sur-traités.
    """)
else:
    st.markdown("""
    ### Attribut Sensible : Zone Géographique (Rural/Urban)
    
    **Pourquoi c'est problématique ?**
    
    Les disparités géographiques dans l'accès aux soins de santé sont un problème majeur :
    - Les zones rurales ont souvent moins d'accès aux établissements de santé spécialisés
    - Les données peuvent être biaisées si elles proviennent principalement de zones urbaines
    - Un modèle biaisé pourrait perpétuer ces inégalités en sous-estimant les risques en zone rurale
    
    **Impact réel** : Un biais géographique pourrait signifier que les patients ruraux à risque élevé ne seraient pas identifiés correctement, aggravant les disparités d'accès aux soins déjà existantes.
    """)

st.markdown("---")

st.subheader("📊 Métriques de Fairness")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Parité Démographique")

    dp_result = demographic_parity_difference(
        y_true=df['stroke'].values,
        y_pred=df['stroke'].values,
        sensitive_attribute=df[sensitive_attr].values
    )

    st.metric(
        "Différence de Parité Démographique",
        f"{dp_result['difference']:.4f}",
        help="Différence maximale entre les taux de prédiction positive par groupe. Plus proche de 0 = plus équitable."
    )

    st.markdown("**Taux par groupe :**")
    for group, rate in dp_result['rates'].items():
        st.write(f"- {group}: {rate:.4f} ({rate*100:.2f}%)")

with col2:
    st.markdown("#### 2. Ratio d'Impact Disproportionné")

    # déterminer dynamiquement les groupes privilégié/non-privilégié basé sur les taux réels
    unique_vals = df[sensitive_attr].unique()
    if len(unique_vals) >= 2:
        # calculer le taux d'AVC pour chaque groupe
        group_rates = {}
        for group in unique_vals:
            group_mask = df[sensitive_attr] == group
            if group_mask.sum() > 0:
                group_rates[group] = df.loc[group_mask, 'stroke'].mean()
        
        if len(group_rates) >= 2:
            # groupe avec le taux le plus élevé = groupe de référence (privilégié)
            # groupe avec le taux le plus faible = groupe comparé (non-privilégié)
            privileged = max(group_rates, key=group_rates.get)
            unprivileged = min(group_rates, key=group_rates.get)
        else:
            st.warning("Pas assez de groupes pour calculer le ratio DI")
            privileged = None
            unprivileged = None
    else:
        st.warning("Pas assez de groupes pour calculer le ratio DI")
        privileged = None
        unprivileged = None

    if privileged and unprivileged:
        di_result = disparate_impact_ratio(
            y_true=df['stroke'].values,
            y_pred=df['stroke'].values,
            sensitive_attribute=df[sensitive_attr].values,
            unprivileged_value=unprivileged,
            privileged_value=privileged
        )

        ratio = di_result['ratio']
        st.metric(
            "Ratio d'Impact Disproportionné",
            f"{ratio:.4f}",
            help="Ratio entre le taux du groupe non-privilégié et celui du groupe privilégié. Proche de 1 = équitable. < 0.8 ou > 1.25 indique un biais."
        )

        st.markdown("**Taux par groupe :**")
        st.write(f"- {unprivileged} (taux le plus faible): {di_result['unprivileged_rate']:.4f} ({di_result['unprivileged_rate']*100:.2f}%)")
        st.write(f"- {privileged} (taux le plus élevé - référence): {di_result['privileged_rate']:.4f} ({di_result['privileged_rate']*100:.2f}%)")

        if ratio < 0.8:
            st.warning(f"⚠️ Biais détecté : Le groupe '{unprivileged}' a un taux significativement plus faible que '{privileged}' (< 0.8)")
        elif ratio > 1.25:
            st.warning(f"⚠️ Biais détecté : Le groupe '{unprivileged}' a un taux significativement plus élevé que '{privileged}' (> 1.25)")
        else:
            st.success("✅ Ratio dans la plage acceptable (0.8 - 1.25)")

st.markdown("---")

st.subheader("📈 Visualisation des Résultats")

if sensitive_attr in df.columns:
    group_rates = df.groupby(sensitive_attr)['stroke'].mean().reset_index()
    group_rates.columns = ['Groupe', 'Taux_AVC']
    group_rates['Taux_AVC_Pourcent'] = group_rates['Taux_AVC'] * 100

    fig = px.bar(
        group_rates,
        x='Groupe',
        y='Taux_AVC_Pourcent',
        title=f"Taux d'AVC par {sensitive_attr}",
        labels={'Taux_AVC_Pourcent': "Taux d'AVC (%)", 'Groupe': sensitive_attr},
        color='Groupe',
        text='Taux_AVC_Pourcent'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    group_counts = df.groupby(sensitive_attr)['stroke'].agg(['count', 'sum']).reset_index()
    group_counts.columns = ['Groupe', 'Total', 'AVC']
    group_counts['Taux'] = (group_counts['AVC'] / group_counts['Total'] * 100).round(2)
    st.dataframe(group_counts, use_container_width=True)

st.markdown("---")

st.subheader("🔬 Comparaison Détaillée par Groupe")

if sensitive_attr in df.columns:
    comparison_df = df.groupby([sensitive_attr, 'stroke']).size().reset_index(name='count')
    comparison_df['stroke_label'] = comparison_df['stroke'].map({0: 'Sans AVC', 1: 'Avec AVC'})

    fig = px.bar(
        comparison_df,
        x=sensitive_attr,
        y='count',
        color='stroke_label',
        title=f"Distribution des AVC par {sensitive_attr}",
        labels={'count': 'Nombre de patients'},
        color_discrete_map={'Sans AVC': '#4ECDC4', 'Avec AVC': '#FF6B6B'},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("💡 Interprétation")

if sensitive_attr == 'gender':
    st.markdown("""
    **Que signifie concrètement le biais détecté ?**
    
    Les métriques calculées révèlent les différences dans les taux d'AVC observés entre les groupes de genre.
    Une différence de parité démographique élevée ou un ratio d'impact disproportionné éloigné de 1
    indique que les taux d'AVC ne sont pas équitablement répartis entre les différents genres.
    
    **Quel groupe est défavorisé ?**
    
    Le groupe avec le taux d'AVC le plus faible pourrait être sous-diagnostiqué ou avoir des facteurs
    de risque non pris en compte. Le groupe avec le taux le plus élevé pourrait avoir des facteurs
    de risque spécifiques ou bénéficier d'un meilleur accès au diagnostic.
    
    **Quel serait l'impact réel de ce biais ?**
    
    Si un modèle de prédiction reproduit ces disparités sans les comprendre, il pourrait :
    - Sous-estimer le risque pour certains groupes, retardant les interventions préventives
    - Sur-estimer le risque pour d'autres groupes, entraînant des traitements inutiles
    - Perpétuer les inégalités existantes dans l'accès aux soins
    
    **Recommandations pour réduire le biais :**
    
    1. **Collecte de données équilibrée** : S'assurer que le dataset contient une représentation équitable de tous les genres
    2. **Analyse par sous-groupes** : Développer des modèles spécifiques ou ajuster les seuils de décision par groupe
    3. **Validation continue** : Surveiller régulièrement les performances du modèle par groupe démographique
    4. **Transparence** : Documenter clairement les limitations et biais potentiels du modèle
    """)
else:
    st.markdown("""
    **Que signifie concrètement le biais détecté ?**
    
    Les métriques révèlent les différences dans les taux d'AVC entre les zones rurales et urbaines.
    Ces différences peuvent refléter à la fois des disparités réelles dans la santé et des biais
    dans la collecte de données ou l'accès aux soins.
    
    **Quel groupe est défavorisé ?**
    
    Les zones rurales sont souvent défavorisées en termes d'accès aux soins spécialisés et aux
    technologies médicales avancées. Un taux d'AVC différentiel pourrait indiquer :
    - Des différences réelles dans les facteurs de risque (alimentation, activité physique, etc.)
    - Des différences dans l'accès au diagnostic et au traitement
    - Des biais dans la collecte de données
    
    **Quel serait l'impact réel de ce biais ?**
    
    Un modèle biaisé géographiquement pourrait :
    - Ignorer les besoins spécifiques des populations rurales
    - Perpétuer les inégalités d'accès aux soins
    - Ne pas tenir compte des facteurs environnementaux spécifiques à chaque zone
    
    **Recommandations pour réduire le biais :**
    
    1. **Données représentatives** : Inclure des données provenant de zones rurales et urbaines de manière équilibrée
    2. **Facteurs contextuels** : Intégrer des variables spécifiques à chaque zone (accès aux soins, distance aux hôpitaux, etc.)
    3. **Modèles adaptatifs** : Développer des modèles qui s'adaptent aux contextes géographiques
    4. **Équité géographique** : S'assurer que les interventions médicales sont accessibles à tous, indépendamment de la localisation
    """)

st.markdown("---")
st.subheader("📋 Résumé des Métriques")

metrics_summary = pd.DataFrame({
    'Métrique': [
        'Différence de Parité Démographique',
        'Ratio d\'Impact Disproportionné'
    ],
    'Valeur': [
        f"{dp_result['difference']:.4f}",
        f"{di_result['ratio']:.4f}" if privileged and unprivileged else "N/A"
    ],
    'Interprétation': [
        'Plus proche de 0 = plus équitable',
        'Entre 0.8 et 1.25 = acceptable'
    ]
})

st.dataframe(metrics_summary, use_container_width=True)
