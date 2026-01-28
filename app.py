import streamlit as st
import pandas as pd
import kagglehub
from pathlib import Path

st.set_page_config(
    page_title="Détection de Biais - Prédiction d'AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_dataset():
    try:
        path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
        csv_files = list(Path(path).glob("*.csv"))
        if not csv_files:
            st.error("Aucun fichier CSV trouvé dans le dataset téléchargé")
            return None, None

        df = pd.read_csv(csv_files[0])

        # normalize age column to int
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            median_age = df["age"].median()
            df["age"] = df["age"].fillna(median_age)
            df["age"] = df["age"].round().astype(int)

        return df, path
    except Exception as e:
        st.error(f"Erreur lors du téléchargement ou du nettoyage du dataset: {e}")
        return None, None

df, dataset_path = load_dataset()

if df is None:
    st.error("Impossible de charger le dataset. Veuillez vérifier votre connexion et vos identifiants Kaggle.")
else:
    st.session_state["df"] = df
    st.session_state["dataset_path"] = dataset_path
    st.title("🏠 Accueil")
    st.markdown("---")

    st.header("🧠 Détection de Biais dans le Stroke Prediction Dataset")

    st.subheader("Titre et présentation du dataset")
    st.markdown(
        """
        Le **Stroke Prediction Dataset** (Kaggle) contient des informations démographiques
        et médicales sur des patients, ainsi qu'un indicateur binaire indiquant s'ils
        ont subi un AVC (`stroke`).
        """
    )

    st.subheader("Contexte et problématique")
    st.markdown(
        """
        Les accidents vasculaires cérébraux (AVC) sont une cause majeure de mortalité
        et de handicap. Pouvoir **anticiper le risque d'AVC** à partir de données
        cliniques et démographiques permettrait de cibler plus tôt les patients à risque
        et de proposer des actions préventives.

        Cependant, des **biais** peuvent apparaître dans les données ou dans les modèles
        de prédiction, par exemple selon le **genre** ou la **zone géographique**
        (rural / urbain). L'objectif de cette application est donc **double** :
        explorer le dataset et **détecter d'éventuels biais** dans ces dimensions.
        """
    )
