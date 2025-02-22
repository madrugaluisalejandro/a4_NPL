import streamlit as st
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Texto", layout="wide")

# Cargar modelo spaCy
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("es_core_news_sm")
    except Exception as e:
        st.error("Error: Es necesario instalar el modelo de spaCy para español")
        st.code("python -m spacy download es_core_news_sm")
        raise e

# Cargar modelo de resumen
@st.cache_resource
def load_summary_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        truncation=True
    )

    if torch.cuda.is_available():
        device = 0
        device_map = "auto"
    else:
        device = -1
        device_map = None  # Puedes omitir este parámetro si prefieres

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device_map
    )

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return summarizer


# Cargar clasificador de emociones
@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

# Función para tokenizar
def spacy_tokenizer(text):
    nlp = load_spacy_model()
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "VERB"]]

# Cargar base de conocimientos
base_conocimientos = {
    "datos_relevantes": {
        "temperatura_global": "+1.2°C desde la era preindustrial",
        "nivel_mar": "Incremento de 3.4 mm por año en promedio",
        "emisiones_co2": "36.3 gigatoneladas en 2021"
    }
}

class AnalizadorTexto:
    def __init__(self):
        self.emotion_classifier = load_emotion_classifier()
        self.vectorizer = TfidfVectorizer(
            analyzer=spacy_tokenizer,
            max_features=20,
            ngram_range=(1,2)
        )
        self.summarizer = load_summary_model()

    def obtener_resumen(self, texto, max_length=300):
        try:
            summary = self.summarizer(
                texto,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            st.error(f"Error en la generación del resumen: {str(e)}")
            return ""

    def extraer_palabras_clave(self, texto):
        try:
            tfidf_matrix = self.vectorizer.fit_transform([texto])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            palabra_score = dict(zip(feature_names, tfidf_scores))
            return dict(sorted(palabra_score.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            st.error(f"Error en extracción de palabras clave: {str(e)}")
            return {}

    def clasificar_emociones(self, texto):
        try:
            resultados = self.emotion_classifier(texto)
            if len(resultados) > 0:
                return {item['label']: item['score'] for item in resultados[0]}
            return {}
        except Exception as e:
            st.error(f"Error en clasificación de emociones: {str(e)}")
            return {}

    def procesar_texto(self, texto):
        resumen = self.obtener_resumen(texto)
        palabras_clave = self.extraer_palabras_clave(texto)
        emociones = self.clasificar_emociones(texto)
        emocion_predominante = max(emociones.items(), key=lambda x: x[1])[0] if emociones else None
        
        return {
            "resumen": resumen,
            "palabras_clave": palabras_clave,
            "emociones": emociones,
            "emocion_predominante": emocion_predominante
        }

def main():
    st.title("Analizador de Texto")
    
    # Inicialización del estado de la sesión
    if 'analizador' not in st.session_state:
        st.session_state.analizador = AnalizadorTexto()

    # Área de entrada de texto
    texto_usuario = st.text_area(
        "Ingrese el texto a analizar:",
        height=200
    )
    
    if st.button("Analizar Texto") and texto_usuario:
        with st.spinner("Analizando texto..."):
            resultados = st.session_state.analizador.procesar_texto(texto_usuario)
            
            # Mostrar resumen
            st.subheader("Resumen")
            st.write(resultados["resumen"])
            
            # Mostrar resultados en columnas
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
            st.subheader("Conocimiento Relevante")
            for key, value in base_conocimientos["datos_relevantes"].items():
                st.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
