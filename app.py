
# =========================================
# 0. Importación de librerías y configuración
# =========================================

import torch
import gc
import re
import json
import graphviz
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load("es_core_news_sm")
except Exception as e:
    print("Instala el modelo spaCy para español: !python -m spacy download es_core_news_sm")
    raise e

# Función para tokenizar utilizando spaCy (sólo sustantivos y verbos)
def spacy_tokenizer(text):
    doc = nlp(text)
    # Filtramos únicamente sustantivos y verbos
    return [token.text for token in doc if token.pos_ in ["NOUN", "VERB"]]


# =========================================
# 1. SISTEMA COGNITIVO (Resumen y Conocimiento Explícito)
# =========================================

base_conocimientos_json = """
{
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
"""
base_conocimientos = json.loads(base_conocimientos_json)

class ModeloResumenOptimizado:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"  # Se puede cambiar por otro modelo
        self.setup_model()

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1024,
            truncation=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def obtener_resumen(self, texto, temperature=0.7, max_length=150):
        try:
            summary = self.summarizer(
                texto,
                max_length=max_length,
                min_length=30,
                temperature=temperature,
                do_sample=False,
                truncation=True
            )
            torch.cuda.empty_cache()
            gc.collect()
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error en la generación del resumen: {str(e)}")
            return ""

class SistemaCognitivo:
    def __init__(self):
        self.modelo_resumen = ModeloResumenOptimizado()
        self.conocimiento_explicito = base_conocimientos
        self.historial = []  # Para almacenar interacciones

    def nodo_percepcion_atencion(self, texto_usuario):
        return texto_usuario

    def nodo_base_conocimientos(self):
        return self.conocimiento_explicito.get("datos_relevantes", {})

    def nodo_razonamiento_llm(self, texto):
        return self.modelo_resumen.obtener_resumen(texto)

    def nodo_memoria_integracion(self, texto_resumen, datos_explicitos):
        integracion = (
            f"Resumen (resultado del modelo):\n{texto_resumen}\n\n"
            "Conocimiento Explícito:\n"
        )
        for key, value in datos_explicitos.items():
            integracion += f"  • {key}: {value}\n"
        integracion += "\nIntegración: El resumen complementa y contrasta la información explícita."
        self.historial.append(integracion)
        return integracion

    def procesar_texto(self, texto_usuario):
        texto_relevante = self.nodo_percepcion_atencion(texto_usuario)
        datos_explicitos = self.nodo_base_conocimientos()
        resumen_modelo = self.nodo_razonamiento_llm(texto_relevante)
        salida_final = self.nodo_memoria_integracion(resumen_modelo, datos_explicitos)
        return salida_final


# =========================================
# 2. ANALIZADOR DE TEXTO (Keywords y Emociones)
# =========================================

class AnalizadorTexto:
    def __init__(self):
        # Pipeline para clasificación de emociones
        self.emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        # Vectorizador TF-IDF utilizando nuestro tokenizer personalizado (solo sustantivos y verbos)
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
        palabra_score_ordenado = dict(sorted(palabra_score.items(), key=lambda x: x[1], reverse=True))
        return palabra_score_ordenado

    def clasificar_emociones(self, texto):
        resultados = self.emotion_classifier(texto)
        dict_emociones = {}
        if len(resultados) > 0:
            for item in resultados[0]:
                dict_emociones[item['label']] = item['score']
        return dict_emociones

    def procesar_texto(self, texto):
        # Limpieza básica
        texto_limpio = re.sub(r"http\S+|www\S+|https\S+", '', texto, flags=re.MULTILINE)
        texto_limpio = re.sub(r"[^\w\s]", '', texto_limpio)
        texto_limpio = texto_limpio.lower().strip()

        # Extraer palabras clave (sólo sustantivos y verbos)
        palabras_clave = self.extraer_palabras_clave(texto_limpio)

        # Clasificar emociones
        emociones = self.clasificar_emociones(texto_limpio)

        # Determinar emoción predominante
        emocion_predominante = None
        max_score = 0.0
        for emo, sc in emociones.items():
            if sc > max_score:
                max_score = sc
                emocion_predominante = emo

        resultado = {
            "palabras_clave": palabras_clave,
            "emociones": emociones,
            "emocion_predominante": emocion_predominante
        }
        return resultado

    def graficar_palabras_clave(self, palabras_clave):
        if not palabras_clave:
            print("No hay palabras clave para graficar.")
            return
        top_items = list(palabras_clave.items())[:10]
        palabras = [x[0] for x in top_items]
        scores = [x[1] for x in top_items]
        plt.figure(figsize=(8,4))
        plt.barh(palabras[::-1], scores[::-1], color='skyblue')
        plt.xlabel("Importancia TF-IDF")
        plt.title("Principales Palabras Clave (sustantivos y verbos)")
        plt.show()

    def graficar_emociones(self, emociones):
        if not emociones:
            print("No se detectaron emociones.")
            return
        total_score = sum(emociones.values())
        if total_score == 0:
            print("No se detectaron emociones con puntuación > 0.")
            return
        emociones_porcent = {k: (v/total_score)*100 for k,v in emociones.items()}
        emociones_ordenadas = dict(sorted(emociones_porcent.items(), key=lambda x: x[1], reverse=True))
        labels = list(emociones_ordenadas.keys())
        values = list(emociones_ordenadas.values())
        plt.figure(figsize=(8,4))
        plt.bar(labels, values, color='salmon')
        plt.ylabel("Porcentaje (%)")
        plt.title("Distribución de Emociones")
        plt.show()


# =========================================
# 3. GENERACIÓN DEL MAPA CONCEPTUAL (Mermaid)
# =========================================

def generar_mapa_conceptual(texto_original, resumen, analisis):
    # Obtener las primeras 5 palabras clave de forma segura
    keywords = list(analisis["palabras_clave"].keys())[:5]
    keywords_str = ", ".join(keywords) if keywords else "No disponibles"

    # Obtener la emoción predominante de forma segura
    emocion = analisis.get('emocion_predominante', 'No detectada')

    # Generar el diagrama evitando caracteres problemáticos
    mermaid_diagram = f"""flowchart TD
    A[Texto Original]
    B[Resumen]
    C[Conocimiento Explícito]
    D[Palabras Clave: {keywords_str}]
    E[Emoción: {emocion}]
    F[Lógica del Problema]

    A --> B
    B --> C
    B --> D
    D --> E
    C --> F
    D --> F
    E --> F"""

    return mermaid_diagram

def generar_imagen_mapa_conceptual(texto_resumido):
    """
    Genera un mapa conceptual simple a partir del resumen del texto.
    Para este ejemplo, se divide el resumen en oraciones y se crean nodos conectados en secuencia.
    """
    dot = graphviz.Digraph(comment='Mapa Conceptual del Texto')

    # Dividir el resumen en oraciones (se asume que están separadas por punto)
    oraciones = [oracion.strip() for oracion in texto_resumido.split('.') if oracion.strip()]

    # Tomamos hasta 4 oraciones para simplificar el mapa conceptual
    nodos = oraciones[:4]

    # Crear nodos para cada oración
    for i, nodo in enumerate(nodos):
        dot.node(f'N{i}', nodo)

    # Conectar los nodos de forma secuencial
    for i in range(len(nodos) - 1):
        dot.edge(f'N{i}', f'N{i+1}')

    return dot
#=========================================
#4. VISUALIZACIÓN DEL CONOCIMIENTO EXPLÍCITO (Recuadros HTML)
#=========================================
def display_conocimiento_explicito(datos_explicitos):
    html = "<div style='border:1px solid #ccc; padding: 10px; margin: 5px;'>"
    html += "<h4>Conocimiento Explícito</h4>"
    for key, value in datos_explicitos.items():
        html += f"<div style='border:1px solid #eee; margin: 5px; padding: 5px; border-radius: 5px;'>"
        html += f"<b>{key.capitalize()}:</b> {value}"
        html += "</div>"
    html += "</div>"
    display(HTML(html))

#=========================================
#5. SISTEMA UNIFICADO
#=========================================
class SistemaUnificado:
    def __init__(self):
        self.sistema_cognitivo = SistemaCognitivo()
        self.analizador_texto = AnalizadorTexto()

    def procesar_texto(self, texto):
        # Se obtiene el resultado del sistema cognitivo (resumen + conocimiento explícito integrado)
        resultado_cognitivo = self.sistema_cognitivo.procesar_texto(texto)
        # Se analiza el resultado (se puede analizar también el texto original si se prefiere)
        analisis = self.analizador_texto.procesar_texto(resultado_cognitivo)
        return resultado_cognitivo, analisis

#=========================================
#6. INTERFAZ UNIFICADA CON WIDGETS
#=========================================
class InterfazUnificada:
    def __init__(self):
        self.sistema = SistemaUnificado()
        self.text_area = widgets.Textarea(
            placeholder="Pega aquí el texto largo (ej. mensajes de X.com) para resumir y analizar...",
            layout=widgets.Layout(width="100%", height="200px")
        )
        self.boton_procesar = widgets.Button(
            description="Procesar Texto",
            button_style="success"
        )
        self.boton_procesar.on_click(self.on_procesar_click)

        # Áreas de salida para mostrar el resumen y el conocimiento explícito
        self.salida_resumen = widgets.Textarea(
            value="",
            placeholder="Aquí aparecerá el resumen...",
            disabled=True,
            layout=widgets.Layout(width="100%", height="120px")
        )
        self.salida_conocimiento = widgets.Output()

        # Área para el mapa conceptual
        self.salida_mapa = widgets.Output()

    def mostrar_ui(self):
        display(self.text_area, self.boton_procesar)
        display(widgets.HTML("<hr><h3>Resumen del Sistema Cognitivo</h3>"))
        display(self.salida_resumen)
        display(widgets.HTML("<hr><h3>Conocimiento Explícito</h3>"))
        display(self.salida_conocimiento)
        display(widgets.HTML("<hr><h3>Mapa Conceptual</h3>"))
        display(self.salida_mapa)

    def on_procesar_click(self, b):
        texto = self.text_area.value.strip()
        if not texto:
            print("Por favor, ingresa texto.")
            return

        # 1. Procesar el texto mediante el sistema unificado
        resultado_cognitivo, analisis = self.sistema.procesar_texto(texto)

        # 2. Actualizar el recuadro del resumen
        self.salida_resumen.value = resultado_cognitivo

        # 3. Mostrar el conocimiento explícito en recuadros (usando HTML)
        with self.salida_conocimiento:
            self.salida_conocimiento.clear_output()
            datos_explicitos = self.sistema.sistema_cognitivo.nodo_base_conocimientos()
            display_conocimiento_explicito(datos_explicitos)

        # 4. Imprimir por consola las palabras clave y emociones
        print("\n>> Palabras clave (Top TF-IDF):")
        for i, (pal, sc) in enumerate(list(analisis["palabras_clave"].items())[:10], start=1):
            print(f"  {i}. {pal} (score={sc:.4f})")
        print("\n>> Emociones detectadas:")
        total_score = sum(analisis["emociones"].values()) if analisis["emociones"] else 0
        if total_score > 0:
            for emo, sc in analisis["emociones"].items():
                print(f"  - {emo}: {sc/total_score*100:.2f}%")
        else:
            print("  No se detectaron emociones.")
        print(f"\n>> Emoción predominante: {analisis['emocion_predominante']}")

        # 5. Graficar palabras clave y emociones
        self.sistema.analizador_texto.graficar_palabras_clave(analisis["palabras_clave"])
        self.sistema.analizador_texto.graficar_emociones(analisis["emociones"])

        # 6. Generar y mostrar el mapa conceptual (Mermaid)
        mapa = generar_mapa_conceptual(texto, self.salida_resumen.value, analisis)
        with self.salida_mapa:
            self.salida_mapa.clear_output()
            display(Markdown(mapa))

        torch.cuda.empty_cache()
        gc.collect()
        mapa_conceptual = generar_imagen_mapa_conceptual(resultado_cognitivo)
        display(mapa_conceptual)

#=========================================
#7. FUNCIÓN PRINCIPAL PARA INICIAR LA INTERFAZ
#=========================================
def iniciar_sistema_unificado():
  interfaz = InterfazUnificada()
  interfaz.mostrar_ui()
  print("Interfaz unificada lista. Pega tu texto y haz clic en 'Procesar Texto'.")

iniciar_sistema_unificado()
