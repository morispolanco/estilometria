import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Configuración de la página
st.set_page_config(page_title="Análisis Estilométrico Forense", layout="wide")

# Función de preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # Tokenización
    words = word_tokenize(text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return words

# Funciones de análisis estilométrico
def basic_stats(text):
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    stats = {
        'Caracteres': len(text),
        'Palabras': len(words),
        'Oraciones': len(sentences),
        'Palabras únicas': len(set(words)),
        'Longitud promedio de palabra': np.mean([len(word) for word in words]),
        'Longitud promedio de oración': np.mean([len(word_tokenize(sent)) for sent in sentences]),
        'Riqueza léxica': len(set(words)) / len(words) * 100
    }
    return stats

def word_frequency(words, n=10):
    return Counter(words).most_common(n)

def punctuation_analysis(text):
    punctuation_marks = ['.', ',', ';', ':', '!', '?', '...', '"', '-', '(', ')']
    counts = {mark: text.count(mark) for mark in punctuation_marks}
    total = sum(counts.values())
    percentages = {mark: (count/total)*100 if total > 0 else 0 for mark, count in counts.items()}
    return counts, percentages

def forensic_report(text, author_name="Desconocido"):
    words = preprocess_text(text)
    stats = basic_stats(text)
    freq_words = word_frequency(words)
    punct_counts, punct_percent = punctuation_analysis(text)
    
    report = f"""
    INFORME FORENSE ESTILOMÉTRICO
    ============================
    Autor analizado: {author_name}
    
    ESTADÍSTICAS BÁSICAS:
    - Caracteres totales: {stats['Caracteres']}
    - Palabras totales: {stats['Palabras']}
    - Oraciones totales: {stats['Oraciones']}
    - Palabras únicas: {stats['Palabras únicas']}
    - Longitud promedio de palabra: {stats['Longitud promedio de palabra']:.2f} caracteres
    - Longitud promedio de oración: {stats['Longitud promedio de oración']:.2f} palabras
    - Riqueza léxica: {stats['Riqueza léxica']:.2f}%
    
    PALABRAS MÁS FRECUENTES:
    {', '.join([f"{word} ({count})" for word, count in freq_words])}
    
    ANÁLISIS DE PUNTUACIÓN:
    """
    for mark, percent in punct_percent.items():
        report += f"- {mark}: {percent:.2f}%\n"
    
    return report

# Interfaz de usuario
st.title("🔍 Análisis Estilométrico Forense")
st.markdown("Herramienta para el análisis de estilo de escritura con aplicaciones forenses")

# Sidebar para configuración
st.sidebar.header("Configuración")
author_name = st.sidebar.text_input("Nombre del autor (opcional)", "Desconocido")

# Área de entrada de texto
st.subheader("Ingrese el texto a analizar")
text_input = st.text_area("Pegue el texto aquí o cargue un archivo", height=200)

# Opción para cargar archivo
uploaded_file = st.file_uploader("O cargue un archivo de texto (.txt)", type=["txt"])

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text_input = stringio.read()

# Botón de análisis
if st.button("Realizar Análisis") and text_input:
    # Preprocesamiento
    words = preprocess_text(text_input)
    
    # Análisis básico
    stats = basic_stats(text_input)
    
    # Mostrar estadísticas básicas
    st.subheader("Estadísticas Básicas")
    col1, col2, col3 = st.columns(3)
    col1.metric("Palabras totales", stats['Palabras'])
    col2.metric("Oraciones", stats['Oraciones'])
    col3.metric("Riqueza léxica", f"{stats['Riqueza léxica']:.2f}%")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Longitud promedio palabra", f"{stats['Longitud promedio de palabra']:.2f}")
    col5.metric("Longitud promedio oración", f"{stats['Longitud promedio de oración']:.2f}")
    col6.metric("Palabras únicas", stats['Palabras únicas'])
    
    # Análisis de frecuencia de palabras
    st.subheader("Palabras más frecuentes")
    freq_words = word_frequency(words)
    df_freq = pd.DataFrame(freq_words, columns=['Palabra', 'Frecuencia'])
    st.bar_chart(df_freq.set_index('Palabra'))
    
    # Análisis de puntuación
    st.subheader("Análisis de Puntuación")
    punct_counts, punct_percent = punctuation_analysis(text_input)
    df_punct = pd.DataFrame({
        'Signo': list(punct_percent.keys()),
        'Porcentaje': list(punct_percent.values())
    })
    st.bar_chart(df_punct.set_index('Signo'))
    
    # Generar informe forense
    report = forensic_report(text_input, author_name)
    
    # Mostrar informe
    st.subheader("Informe Forense")
    st.text_area("Informe completo", report, height=300)
    
    # Botón de descarga
    st.subheader("Descargar Informe")
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="informe_forense.txt">Descargar informe como archivo de texto</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Análisis comparativo (si hay múltiples textos)
    st.subheader("Análisis Comparativo (experimental)")
    st.info("Para análisis comparativo, cargue múltiples textos usando la opción de archivo")
    
    if uploaded_file is not None:
        # Aquí podrías agregar lógica para comparar con otros textos
        st.write("Funcionalidad de comparación en desarrollo")

# Instrucciones
with st.expander("Instrucciones de uso"):
    st.markdown("""
    1. **Ingrese el texto**: Pegue el texto directamente en el área de texto o cargue un archivo .txt
    2. **Configure el análisis**: Opcionalmente, ingrese el nombre del autor
    3. **Realice el análisis**: Haga clic en el botón "Realizar Análisis"
    4. **Explore los resultados**:
        - Estadísticas básicas del texto
        - Gráficos de frecuencia de palabras
        - Análisis de patrones de puntuación
        - Informe forense completo
    5. **Descargue el informe**: Obtenga un archivo .txt con el informe completo
    """)

# Notas técnicas
with st.expander("Notas técnicas"):
    st.markdown("""
    - **Preprocesamiento**: Se eliminan caracteres especiales, números y stopwords
    - **Riqueza léxica**: Calculada como (palabras únicas / palabras totales) * 100
    - **Análisis de puntuación**: Se consideran los signos más comunes en español
    - **Limitaciones**: El análisis es básico y no incluye análisis sintáctico profundo
    - **Aplicaciones forenses**: Útil para comparación de estilos de escritura, detección de plagio, atribución de autoría
    """)
