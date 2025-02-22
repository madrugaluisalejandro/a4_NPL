import streamlit as st
import spacy
import json
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Texto", layout="wide")

# Cargar el modelo de spaCy
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("es_core_news_sm")
    except Exception as e:
        st.error("Por favor, instala el modelo de spaCy: python -m spacy download es_core_news_sm")
        raise e

nlp = load_spacy_model()

# Cargar el modelo de emociones
@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

# Base de conocimientos
base_conocimientos = {
    "informacion_general": {
        "titulo": "Noticias sobre cambio climático",
        "fuentes": ["IPCC", "UNEP", "NASA"],
        "definicion": "El cambio climático se refiere a cambios de largo plazo en los patrones de temperatura y clima."
    },
    "datos_relevantes": {
        "temperatura_global": "+1.2°C desde la era preindustrial",
        "nivel_mar": "Incremento de 3.4 mm por año en promedio",
        "emisiones_co2": "36.3 gigatoneladas en 2021"
    }
}

# Función para tokenizar usando spaCy
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "VERB"]]

# Clase para análisis de texto
class AnalizadorTexto:
    def __init__(self):
        self.emotion_classifier = load_emotion_classifier()
        self.vectorizer = TfidfVectorizer(
            analyzer=spacy_tokenizer,
            max_features=20,
            ngram_range=(1,2)
        )
    
    def extraer_palabras_clave(self, texto):
        tfidf_matrix = self.vectorizer.fit_transform([texto])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        palabra_score = dict(zip(feature_names, tfidf_scores))
        return dict(sorted(palabra_score.items(), key=lambda x: x[1], reverse=True))
    
    def clasificar_emociones(self, texto):
        resultados = self.emotion_classifier(texto)
        if len(resultados) > 0:
            return {item['label']: item['score'] for item in resultados[0]}
        return {}

    def procesar_texto(self, texto):
        palabras_clave = self.extraer_palabras_clave(texto)
        emociones = self.clasificar_emociones(texto)
        emocion_predominante = max(emociones.items(), key=lambda x: x[1])[0] if emociones else None
        
        return {
            "palabras_clave": palabras_clave,
            "emociones": emociones,
            "emocion_predominante": emocion_predominante
        }

def main():
    st.title("Analizador de Texto")
    
    # Área de entrada de texto
    texto_usuario = st.text_area(
        "Ingrese el texto a analizar:",
        height=200
    )
    
    if st.button("Analizar Texto") and texto_usuario:
        analizador = AnalizadorTexto()
        
        with st.spinner("Analizando texto..."):
            resultados = analizador.procesar_texto(texto_usuario)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Palabras Clave")
                for palabra, score in list(resultados["palabras_clave"].items())[:10]:
                    st.write(f"- {palabra}: {score:.4f}")
            
            with col2:
                st.subheader("Análisis de Emociones")
                for emocion, score in resultados["emociones"].items():
                    st.write(f"- {emocion}: {score*100:.2f}%")
                
                st.write(f"**Emoción predominante:** {resultados['emocion_predominante']}")
            
            # Mostrar conocimiento explícito
            st.subheader("Conocimiento Explícito")
            for key, value in base_conocimientos["datos_relevantes"].items():
                st.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
