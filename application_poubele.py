# app_streamlit_trash.py
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# === CONFIGURATION ===
# Utiliser un chemin relatif au lieu d'un chemin absolu
MODEL_PATH = "modele_poubelle.h5"
IMG_SIZE = (150, 150)  # Doit correspondre à l'entraînement

# Fonction pour charger le modèle avec cache
@st.cache_resource
def load_model():
    """Charger le modèle une seule fois"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modèle non trouvé à l'emplacement : {MODEL_PATH}")
        st.info("📋 Assurez-vous que le fichier 'modele_poubelle.h5' est dans votre repository GitHub")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.stop()

# Charger le modèle au démarrage
model = load_model()

# === APPLICATION ===
class TrashMonitoringApp:
    def __init__(self):
        # Utiliser session_state pour persister l'historique
        if 'historique' not in st.session_state:
            st.session_state.historique = []

    @property
    def historique(self):
        return st.session_state.historique

    def preprocess_image(self, img_pil):
        """Prétraiter l'image pour la rendre compatible avec le modèle"""
        # Redimensionner exactement comme pendant l'entraînement
        img_resized = img_pil.resize(IMG_SIZE, Image.LANCZOS)
        
        # Convertir en array numpy
        img_array = image.img_to_array(img_resized)
        
        # Normaliser comme pendant l'entraînement (0-1)
        img_array = img_array / 255.0
        
        # Ajouter la dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict_image(self, img_pil):
        """Prédire si l'image contient une poubelle vide ou pleine"""
        try:
            # Prétraiter l'image
            img_array = self.preprocess_image(img_pil)
            
            # Faire la prédiction
            pred = model.predict(img_array, verbose=0)[0][0]
            
            # Interpréter le résultat
            if pred > 0.5:
                return "PLEINE", float(pred)
            else:
                return "VIDE", float(1 - pred)
                
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return "ERREUR", 0.0

    def analyser_et_logger(self, img_pil, localisation=None):
        """Analyser une image et sauvegarder les résultats"""
        statut, confiance = self.predict_image(img_pil)
        
        if statut != "ERREUR":
            entree = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'statut': statut,
                'confiance': round(confiance, 3),
                'localisation': localisation,
            }
            st.session_state.historique.append(entree)
            
        return statut, confiance

    def generer_rapport(self):
        if not self.historique:
            return "Aucune analyse effectuée"
        
        df = pd.DataFrame(self.historique)
        poubelles_pleines = len(df[df['statut'] == 'PLEINE'])
        taux_remplissage = (poubelles_pleines / len(df)) * 100
        
        return f"""
📊 RAPPORT D'ANALYSE
- Total analyses: {len(df)}
- Poubelles pleines: {poubelles_pleines}
- Poubelles vides: {len(df) - poubelles_pleines}
- Taux de remplissage: {taux_remplissage:.1f}%
- Dernière analyse: {df.iloc[-1]['date']}
- Localisation: {df.iloc[-1]['localisation']}
"""

# === INTERFACE STREAMLIT ===
def main():
    st.set_page_config(page_title="Suivi des déchets App", layout="wide")
    st.title("🗑️ Surveillance Intelligente des déchets ")
    
    # Information sur le modèle
    st.sidebar.subheader("📋 Informations modèle")
    st.sidebar.success("✅ Modèle chargé avec succès!")
    st.sidebar.write(f"Taille d'image: {IMG_SIZE}")
    st.sidebar.write(f"Classes: VIDE/PLEINE")
    
    # Bouton pour télécharger le modèle
    st.sidebar.subheader("📥 Téléchargement du modèle")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            st.sidebar.download_button(
                label="📥 Télécharger le modèle",
                data=file,
                file_name="modele_poubelle.h5",
                mime="application/octet-stream"
            )

    app = TrashMonitoringApp()

    mode = st.sidebar.selectbox("Mode d'analyse", ["📷 Caméra/Upload", "📈 Rapports"])

    if mode == "📷 Caméra/Upload":
        st.header("Analyse d'image de poubelle")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Capture d'image")
            uploaded_file = st.camera_input("Prenez une photo de la poubelle", key="cam_input")
            
            if uploaded_file is None:
                st.subheader("📁 Ou uploader une image")
                uploaded_file = st.file_uploader("Choisir une image", type=['jpg','jpeg','png'])

        with col2:
            st.subheader("🏷️ Paramètres")
            localisation = st.text_input("Localisation", "SENEGAL")
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                
                # Afficher l'original et le redimensionné
                st.image(img, caption="Image originale", use_column_width=True)
                
                # Aperçu du redimensionnement
                img_preview = img.resize(IMG_SIZE, Image.LANCZOS)
                st.image(img_preview, caption=f"Image pour le modèle ({IMG_SIZE[0]}x{IMG_SIZE[1]})", width=150)

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            
            if st.button("🔍 Analyser la poubelle", type="primary"):
                with st.spinner("Analyse en cours..."):
                    statut, confiance = app.analyser_et_logger(img, localisation)
                    
                    # Afficher les résultats
                    st.subheader("📊 Résultats de l'analyse")
                    
                    if statut == "PLEINE":
                        st.error(f"🚨 **POUBELLE PLEINE**")
                        st.write(f"**Niveau de confiance:** {confiance:.1%}")
                        st.warning("**📍 Action recommandée:** Programmer la collecte rapidement")
                    elif statut == "VIDE":
                        st.success(f"✅ **POUBELLE VIDE**")
                        st.write(f"**Niveau de confiance:** {confiance:.1%}")
                        st.info("**📋 Statut:** Surveillance normale - Pas d'action nécessaire")
                    else:
                        st.error("❌ **Erreur lors de l'analyse**")

    elif mode == "📈 Rapports":
        st.header("Rapports et Statistiques")
        
        rapport = app.generer_rapport()
        st.text_area("Rapport d'activité", rapport, height=200)

        if app.historique:
            df = pd.DataFrame(app.historique)
            
            st.subheader("📋 Historique détaillé")
            st.dataframe(df, use_container_width=True)
            
            # Statistiques visuelles
            col1, col2, col3 = st.columns(3)
            with col1:
                total = len(df)
                st.metric("Total analyses", total)
            with col2:
                pleines = len(df[df['statut'] == 'PLEINE'])
                st.metric("Poubelles pleines", pleines)
            with col3:
                taux = (pleines / total * 100) if total > 0 else 0
                st.metric("Taux remplissage", f"{taux:.1f}%")
            
            # Option d'export
            st.subheader("💾 Export des données")
            csv = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"rapport_poubelles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Aucune analyse effectuée pour le moment")

if __name__ == "__main__":
    main()