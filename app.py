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
        'Riqueza léxica': len(set(words)) / len(words) * 100 if len(words) > 0 else 0
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

# Función para calcular similitud entre dos textos
def calculate_similarity(text1, text2):
    # Obtener estadísticas básicas
    stats1 = basic_stats(text1)
    stats2 = basic_stats(text2)
    
    # Preprocesar textos
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    # Frecuencia de palabras
    freq1 = dict(word_frequency(words1, 20))
    freq2 = dict(word_frequency(words2, 20))
    
    # Análisis de puntuación
    punct_counts1, punct_percent1 = punctuation_analysis(text1)
    punct_counts2, punct_percent2 = punctuation_analysis(text2)
    
    # Calcular similitudes para cada métrica
    similarities = {}
    
    # Riqueza léxica
    lex_sim = 1 - abs(stats1['Riqueza léxica'] - stats2['Riqueza léxica']) / 100
    similarities['Riqueza léxica'] = lex_sim
    
    # Longitud promedio de palabra
    word_len_sim = 1 - abs(stats1['Longitud promedio de palabra'] - stats2['Longitud promedio de palabra']) / 10
    similarities['Longitud promedio de palabra'] = word_len_sim
    
    # Longitud promedio de oración
    sent_len_sim = 1 - abs(stats1['Longitud promedio de oración'] - stats2['Longitud promedio de oración']) / 30
    similarities['Longitud promedio de oración'] = sent_len_sim
    
    # Frecuencia de palabras
    common_words = set(freq1.keys()) & set(freq2.keys())
    if len(common_words) > 0:
        word_sim = len(common_words) / 20  # Normalizado al top 20
    else:
        word_sim = 0
    similarities['Frecuencia de palabras'] = word_sim
    
    # Uso de puntuación
    punct_sim = 0
    punct_signs = ['.', ',', ';', ':', '!', '?', '...', '"', '-', '(', ')']
    for sign in punct_signs:
        percent1 = punct_percent1.get(sign, 0)
        percent2 = punct_percent2.get(sign, 0)
        punct_sim += 1 - abs(percent1 - percent2) / 100
    punct_sim /= len(punct_signs)
    similarities['Uso de puntuación'] = punct_sim
    
    # Calcular similitud total (promedio ponderado)
    weights = {
        'Riqueza léxica': 0.25,
        'Longitud promedio de palabra': 0.15,
        'Longitud promedio de oración': 0.15,
        'Frecuencia de palabras': 0.3,
        'Uso de puntuación': 0.15
    }
    
    total_similarity = sum(similarities[metric] * weights[metric] for metric in similarities)
    
    return total_similarity, similarities

# Interfaz de usuario
st.title("🔍 Análisis Estilométrico Forense")
st.markdown("Herramienta para el análisis de estilo de escritura con aplicaciones forenses")

# Selector de modo de análisis
analysis_mode = st.sidebar.radio("Modo de análisis", ["Análisis individual", "Comparación de textos"])

if analysis_mode == "Análisis individual":
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

else:  # Modo de comparación
    st.header("Comparación de Textos")
    st.markdown("Ingrese dos textos para determinar si son del mismo autor")
    
    # Dos columnas para los textos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Texto 1")
        text1 = st.text_area("Pegue el primer texto aquí", height=200)
        file1 = st.file_uploader("Cargar archivo para Texto 1", type=["txt"], key="file1")
        
        if file1 is not None:
            stringio = StringIO(file1.getvalue().decode("utf-8"))
            text1 = stringio.read()
    
    with col2:
        st.subheader("Texto 2")
        text2 = st.text_area("Pegue el segundo texto aquí", height=200)
        file2 = st.file_uploader("Cargar archivo para Texto 2", type=["txt"], key="file2")
        
        if file2 is not None:
            stringio = StringIO(file2.getvalue().decode("utf-8"))
            text2 = stringio.read()
    
    # Botón de comparación
    if st.button("Comparar Textos") and text1 and text2:
        # Calcular similitud
        total_similarity, similarities = calculate_similarity(text1, text2)
        
        # Mostrar resultados
        st.subheader("Resultado de la Comparación")
        st.metric("Similitud Estilométrica", f"{total_similarity:.2%}")
        
        # Interpretación del resultado
        if total_similarity >= 0.85:
            st.success("✅ Muy alta probabilidad de que los textos sean del mismo autor")
            conclusion = "Los textos muestran un estilo de escritura muy similar en todas las métricas analizadas."
        elif total_similarity >= 0.70:
            st.info("ℹ️ Alta probabilidad de que los textos sean del mismo autor")
            conclusion = "Los textos comparten características estilísticas significativas."
        elif total_similarity >= 0.50:
            st.warning("⚠️ Posible similitud, pero no concluyente")
            conclusion = "Los textos muestran algunas similitudes, pero también diferencias importantes."
        else:
            st.error("❌ Baja probabilidad de que los textos sean del mismo autor")
            conclusion = "Los textos presentan diferencias significativas en su estilo de escritura."
        
        st.markdown(f"**Conclusión:** {conclusion}")
        
        # Mostrar detalles de similitud
        st.subheader("Análisis Detallado por Métrica")
        df_sim = pd.DataFrame({
            'Métrica': list(similarities.keys()),
            'Similitud': [f"{v:.2%}" for v in similarities.values()]
        })
        st.dataframe(df_sim)
        
        # Gráfico de similitudes
        st.subheader("Gráfico de Similitudes")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=list(similarities.keys()), y=list(similarities.values()), ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similitud")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Comparación de estadísticas básicas
        st.subheader("Comparación de Estadísticas Básicas")
        stats1 = basic_stats(text1)
        stats2 = basic_stats(text2)
        
        df_stats = pd.DataFrame({
            'Métrica': ['Caracteres', 'Palabras', 'Oraciones', 'Palabras únicas', 
                        'Longitud promedio de palabra', 'Longitud promedio de oración', 'Riqueza léxica'],
            'Texto 1': [stats1['Caracteres'], stats1['Palabras'], stats1['Oraciones'], 
                        stats1['Palabras únicas'], f"{stats1['Longitud promedio de palabra']:.2f}", 
                        f"{stats1['Longitud promedio de oración']:.2f}", f"{stats1['Riqueza léxica']:.2f}%"],
            'Texto 2': [stats2['Caracteres'], stats2['Palabras'], stats2['Oraciones'], 
                        stats2['Palabras únicas'], f"{stats2['Longitud promedio de palabra']:.2f}", 
                        f"{stats2['Longitud promedio de oración']:.2f}", f"{stats2['Riqueza léxica']:.2f}%"]
        })
        st.dataframe(df_stats)
        
        # Generar informe comparativo
        report = f"""
        INFORME DE COMPARACIÓN ESTILOMÉTRICA
        ====================================
        
        SIMILITUD TOTAL: {total_similarity:.2%}
        
        CONCLUSIÓN: {conclusion}
        
        ANÁLISIS DETALLADO:
        """
        for metric, sim in similarities.items():
            report += f"- {metric}: {sim:.2%}\n"
        
        report += f"""
        
        ESTADÍSTICAS TEXTO 1:
        - Caracteres: {stats1['Caracteres']}
        - Palabras: {stats1['Palabras']}
        - Oraciones: {stats1['Oraciones']}
        - Palabras únicas: {stats1['Palabras únicas']}
        - Longitud promedio de palabra: {stats1['Longitud promedio de palabra']:.2f}
        - Longitud promedio de oración: {stats1['Longitud promedio de oración']:.2f}
        - Riqueza léxica: {stats1['Riqueza léxica']:.2f}%
        
        ESTADÍSTICAS TEXTO 2:
        - Caracteres: {stats2['Caracteres']}
        - Palabras: {stats2['Palabras']}
        - Oraciones: {stats2['Oraciones']}
        - Palabras únicas: {stats2['Palabras únicas']}
        - Longitud promedio de palabra: {stats2['Longitud promedio de palabra']:.2f}
        - Longitud promedio de oración: {stats2['Longitud promedio de oración']:.2f}
        - Riqueza léxica: {stats2['Riqueza léxica']:.2f}%
        """
        
        # Mostrar informe
        st.subheader("Informe Comparativo")
        st.text_area("Informe completo", report, height=400)
        
        # Botón de descarga
        st.subheader("Descargar Informe")
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="informe_comparativo.txt">Descargar informe como archivo de texto</a>'
        st.markdown(href, unsafe_allow_html=True)

# Instrucciones
with st.expander("Instrucciones de uso"):
    st.markdown("""
    **Análisis Individual:**
    1. Ingrese el texto a analizar
    2. Configure el análisis (opcionalmente ingrese el nombre del autor)
    3. Haga clic en "Realizar Análisis"
    4. Explore los resultados y descargue el informe
    
    **Comparación de Textos:**
    1. Ingrese dos textos para comparar
    2. Haga clic en "Comparar Textos"
    3. Revise la similitud estilométrica y la conclusión
    4. Explore el análisis detallado y descargue el informe
    """)

# Notas técnicas
with st.expander("Notas técnicas"):
    st.markdown("""
    - **Métricas comparadas:** Riqueza léxica, longitud promedio de palabra/oración, frecuencia de palabras y uso de puntuación
    - **Ponderación:** Las métricas tienen diferentes pesos en el cálculo de similitud total
    - **Interpretación de resultados:**
        - ≥85%: Muy alta probabilidad de mismo autor
        - 70-84%: Alta probabilidad de mismo autor
        - 50-69%: Similitud no concluyente
        - <50%: Baja probabilidad de mismo autor
    - **Limitaciones:** El análisis se basa en patrones estilísticos superficiales, no en análisis semántico profundo
    """)
