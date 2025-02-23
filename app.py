# app.py
import streamlit as st
import torch
import spacy
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Texto",
    layout="wide"
)

@st.cache_resource
def load_models():
    nlp = spacy.load("es_core_news_sm")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )
    return nlp, summarizer

def analyze_text(text, nlp, summarizer):
    # Generar resumen
    summary = summarizer(text, max_length=130, min_length=30, truncation=True)[0]['summary_text']
    
    # Extraer palabras clave
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB"] and not token.is_stop]
    
    # Extraer frases clave (máximo 5 palabras)
    key_phrases = []
    for sent in doc.sents:
        words = [token.text for token in sent if not token.is_stop and not token.is_punct]
        if 2 <= len(words) <= 5:
            key_phrases.append(" ".join(words))
    
    return summary, keywords[:10], key_phrases[:6]

def main():
    st.title("Analizador de Texto")
    
    # Cargar modelos
    nlp, summarizer = load_models()
    
    # Área de entrada de texto
    text = st.text_area("Introduce el texto a analizar", height=200)
    
    if st.button("Analizar"):
        if not text:
            st.warning("Por favor, introduce un texto para analizar.")
            return
            
        with st.spinner("Analizando texto..."):
            summary, keywords, phrases = analyze_text(text, nlp, summarizer)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Resumen")
                st.write(summary)
                
                st.subheader("Frases Clave")
                for i, phrase in enumerate(phrases, 1):
                    st.write(f"{i}. {phrase}")
            
            with col2:
                st.subheader("Palabras Clave")
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(keywords))
                ax.barh(y_pos, [1]*len(keywords))
                ax.set_yticks(y_pos)
                ax.set_yticklabels(keywords)
                ax.invert_yaxis()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
