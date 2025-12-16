import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import google.generativeai as genai
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import time
import random

# Desactivar advertencias de Plotly y pandas
warnings.filterwarnings('ignore')

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Análisis Integral de Backlinks con IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 0. CONFIGURACIÓN E INICIALIZACIÓN DE GEMINI IA
# ==============================================================================

MODEL_NAME = "gemini-2.5-pro"

@st.cache_resource
def configure_gemini(api_key):
    """Configura la API de Gemini si la clave es válida."""
    if not api_key:
        return None, False

    try:
        client = genai.Client(api_key=api_key)
        
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        return client, True, generation_config, safety_settings

    except Exception as e:
        st.error(f"⚠️ Error al configurar Gemini: {e}")
        return None, False, None, None

# ==============================================================================
# 1. FUNCIONES AUXILIARES DE PROCESAMIENTO
# ==============================================================================

def extract_domain(url):
    """Extrae dominio de URL"""
    if pd.isna(url):
        return 'Dominio Desconocido'
    
    try:
        url = str(url).lower().strip()
        patterns = [
            r'https?://(?:www\.)?([^/\?:]+)',
            r'www\.([^/\?:]+)',
            r'([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?=/|\?|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                domain = match.group(1)
                domain = re.sub(r'^www\.', '', domain)
                return domain
    except:
        pass
    
    return 'Dominio Desconocido'

def classify_domain(domain):
    """Clasifica el dominio por tipo y calidad."""
    d = domain.lower()
    if any(x in d for x in ['.edu', '.gov', '.mil']): return 'Alta Autoridad', 'Alta'
    elif any(x in d for x in ['news', 'media', 'press', 'journal']): return 'Medios/Noticias', 'Alta'
    elif 'blog' in d or 'medium.com' in d: return 'Blog/Contenido', 'Media'
    elif any(x in d for x in ['forum', 'reddit.com', 'community']): return 'Foro/Comunidad', 'Baja'
    else: return 'Comercial/General', 'Media'

# ==============================================================================
# 2. SISTEMA DE PROCESAMIENTO CON CACHE
# ==============================================================================

@st.cache_data(show_spinner="Cargando y filtrando datos...")
def load_and_filter_data(uploaded_file, link_type, sample_size):
    """Carga, filtra y prepara el dataset."""
    
    file_name = uploaded_file.name
    file_data = uploaded_file.getvalue()
    
    # Lógica de carga (CSV/XLSX)
    try:
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        elif file_name.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(file_data), encoding='utf-8')
            except:
                df = pd.read_csv(io.BytesIO(file_data), encoding='latin-1')
        else:
            st.error("Formato de archivo no soportado. Por favor, usa .csv o .xlsx.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return pd.DataFrame()
        
    # Mapeo y renombramiento de columnas
    column_mapping = {
        'Domain rating': 'DR', 'Domain Rating': 'DR', 'dr': 'DR', 'domain_rating': 'DR',
        'Domain traffic': 'Traffic', 'Traffic': 'Traffic', 'traffic': 'Traffic',
        'Referring page URL': 'URL', 'URL': 'URL', 'url': 'URL', 'Link': 'URL',
        'Nofollow': 'Nofollow', 'nofollow': 'Nofollow',
        'Type': 'Link_Type', 'Link type': 'Link_Type', 'link_type': 'Link_Type', 'Link Type': 'Link_Type',
        'Anchor': 'Anchor_Text', 'Anchor Text': 'Anchor_Text', 'anchor': 'Anchor_Text'
    }
    
    df = df.rename(columns=lambda x: column_mapping.get(x, x))
    
    # Validar columnas mínimas
    required = ['DR', 'Traffic', 'URL']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.warning(f"Columnas clave faltantes: {missing}. El análisis requiere 'DR', 'Traffic' y 'URL'.")
        return pd.DataFrame()
    
    # 1. Filtrar por tipo de enlace
    if link_type and 'Link_Type' in df.columns:
        mask = df['Link_Type'].astype(str).str.contains(link_type, case=False, na=False)
        df = df[mask]
    
    # 2. Aplicar muestreo
    if sample_size and len(df) > sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # 3. Limpiar y clasificar
    df['DR'] = pd.to_numeric(df['DR'], errors='coerce').fillna(0)
    df['Traffic'] = pd.to_numeric(df['Traffic'], errors='coerce').fillna(0)
    df['Nofollow'] = df.get('Nofollow', pd.Series(['No'] * len(df))).astype(str).str.lower().isin(['true', 'yes', '1', 'nofollow'])
    
    df['Domain'] = df['URL'].apply(extract_domain)
    df = df[df['Domain'] != 'Dominio Desconocido']
    
    classification = df['Domain'].apply(classify_domain)
    df['Domain_Type'] = classification.apply(lambda x: x[0])
    df['Domain_Quality'] = classification.apply(lambda x: x[1])
    
    return df

@st.cache_data(show_spinner="Calculando métricas y Score SEO...")
def calculate_metrics(df, total_backlinks_kommo=0):
    """Calcula métricas agregadas por dominio y el Score SEO."""
    
    if df.empty: return pd.DataFrame(), {}
    
    group_cols = ['Domain', 'Domain_Type', 'Domain_Quality']
    agg_dict = {
        'DR': ('DR', 'max'),
        'Traffic': ('Traffic', 'max'),
        'Total_Links': ('URL', 'size'),
        'Nofollow_Count': ('Nofollow', 'sum'),
        'Anchor_Text': ('Anchor_Text', lambda x: ', '.join(x.dropna().astype(str).unique()[:3]))
    }
    
    domains_df = df.groupby(group_cols).agg(**agg_dict).reset_index()
    
    # Cálculo del Score SEO
    def calculate_seo_score(row):
        score = 0
        dr_score = min(row['DR'] / 100, 1.0) * 40
        score += dr_score
        
        traffic_score = min(np.log10(row['Traffic'] + 1) / 5.5, 1.0) * 30
        score += traffic_score
        
        quality_map = {'Alta': 20, 'Media': 15, 'Baja': 10}
        score += quality_map.get(row['Domain_Quality'], 10)
        
        dofollow_rate = (row['Total_Links'] - row['Nofollow_Count']) / row['Total_Links'] if row['Total_Links'] > 0 else 1.0
        score += dofollow_rate * 10
        
        return round(score, 1)
    
    domains_df['SEO_Score'] = domains_df.apply(calculate_seo_score, axis=1)
    
    # Métricas clave
    total_domains = len(domains_df)
    avg_dr = domains_df['DR'].mean()
    avg_seo_score = domains_df['SEO_Score'].mean()
    total_analyzed_links = df['URL'].nunique()
    
    metrics_summary = {
        'total_domains': total_domains,
        'avg_dr': avg_dr,
        'avg_seo_score': avg_seo_score,
        'coverage': (total_analyzed_links / total_backlinks_kommo) * 100 if total_backlinks_kommo > 0 else None,
        'total_analyzed_links': total_analyzed_links
    }
    
    return domains_df, metrics_summary

# ==============================================================================
# 3. ANÁLISIS Y ESTRATEGIA CON IA
# ==============================================================================

@st.cache_data(show_spinner="Analizando patrones de enlaces 'In-Content'...")
def analyze_in_content_patterns(df):
    """Analiza patrones específicos de enlaces in-content."""
    if 'Link_Type' not in df.columns or df.empty: return {}
    
    content_patterns = ['content', 'article', 'post', 'blog', 'editorial', 'text', 'body']
    in_content_mask = df['Link_Type'].astype(str).str.contains('|'.join(content_patterns), case=False, na=False)
    in_content_df = df[in_content_mask]
    
    if in_content_df.empty: return {}
    
    analysis = {
        'total_in_content': len(in_content_df),
        'percentage_of_total': (len(in_content_df) / len(df)) * 100,
        'avg_dr_in_content': in_content_df['DR'].mean(),
        'avg_traffic_in_content': in_content_df['Traffic'].mean(),
        'dofollow_rate': ((len(in_content_df) - in_content_df['Nofollow'].sum()) / len(in_content_df)) * 100 if len(in_content_df) > 0 else 100,
        'domain_types': in_content_df['Domain_Type'].value_counts().head(5).to_dict()
    }
    return analysis

@st.cache_data(show_spinner="🧠 Generando Estrategia Detallada con Gemini...")
def generate_in_content_strategy(client, generation_config, safety_settings, analysis, target_keywords, overall_metrics):
    """Llama a Gemini para generar una estrategia detallada."""
    
    if not client: return "Error: Cliente de IA no configurado."
    
    prompt = f"""
    Eres un Director de Estrategia de Link Building experto con un enfoque creativo y táctico. Genera una estrategia completa y accionable para los enlaces in-content del cliente.

    CONTEXTO DEL ANÁLISIS:
    • Total de enlaces in-content analizados: {analysis.get('total_in_content', 0):,}
    • Representan el {analysis.get('percentage_of_total', 0):.1f}% del perfil total
    • Tasa dofollow en in-content: {analysis.get('dofollow_rate', 0):.1f}%
    • DR promedio de fuentes: {analysis.get('avg_dr_in_content', 0):.1f}
    • Tipos de dominio principales: {json.dumps(analysis.get('domain_types', {}), indent=2)}
    
    OBJETIVOS DEL CLIENTE:
    • Domain Rating objetivo: {overall_metrics.get('dr_total', 'N/A')}
    • Tráfico orgánico mensual objetivo: {overall_metrics.get('traffic_total', 0):,}
    • Palabras clave objetivo: {', '.join(target_keywords) if target_keywords else 'No especificadas'}
    
    GENERA UNA ESTRATEGIA COMPLETA DE MÍNIMO 1500 PALABRAS CON LAS SIGUIENTES SECCIONES EN ESPAÑOL:
    
    1. DIAGNÓSTICO ESTRATÉGICO: Fortalezas, Debilidades y Riesgos de Diversificación.
    2. ESTRATEGIA DE CONTENIDO: 3 Tipos de contenido "link-magnet" y sus temáticas específicas.
    3. TÁCTICAS DE OUTREACH 2.0: Métodos creativos y personalizados para la adquisición de enlaces (no genéricos).
    4. PLAN DE OPTIMIZACIÓN DE BACKLINKS EXISTENTES: Estrategias para aumentar el "link juice" y convertir NoFollow.
    5. KPIs Y MEDICIÓN DE IMPACTO: Metas cuantificables para los próximos 6 meses.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=generation_config,
            safety_settings=safety_settings
        )
        return response.text
        
    except Exception as e:
        return f"Error generando estrategia: {str(e)}"

# ==============================================================================
# 4. VISUALIZACIONES ORGANIZADAS
# ==============================================================================

# Nota: Se utiliza st.plotly_chart para renderizar las figuras de Plotly.

def create_executive_summary(metrics, domains_df):
    """Crea un resumen ejecutivo visual con indicadores (KPIs)."""
    if domains_df.empty: return go.Figure()

    summary_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '1. Score SEO y Autoridad Promedio',
            '2. Distribución de Calidad de Dominios',
            '3. Score SEO por Tipo de Dominio (Top 5)',
            '4. Score SEO por Autoridad (DR Range)'
        ),
        specs=[
            [{'type': 'indicator', 'rowspan': 1}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.15, horizontal_spacing=0.2
    )
    
    avg_seo_score = domains_df['SEO_Score'].mean()
    total_domains = len(domains_df)
    
    # 1. Indicadores clave (Score SEO)
    summary_fig.add_trace(
        go.Indicator(
            mode="number+gauge", value=avg_seo_score,
            title={"text": f"Score SEO Promedio (Dominios Únicos: {total_domains:,})"},
            number={"font": {"size": 40}, "suffix": "/100"},
            gauge={
                'axis': {'range': [0, 100]}, 'bar': {'color': "#4CAF50"},
                'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 80], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 75}
            },
            domain={'row': 0, 'column': 0, 'x':[0, 1.0], 'y':[0, 1.0]}
        ), row=1, col=1
    )
    
    # 2. Distribución de Calidad
    quality_dist = domains_df['Domain_Quality'].value_counts()
    summary_fig.add_trace(
        go.Pie(
            labels=quality_dist.index, values=quality_dist.values, hole=0.4, 
            marker_colors=['#2E8B57', '#FFA500', '#FF6347'], textinfo='percent+label', 
        ), row=1, col=2
    )
    
    # 3. Score SEO por Tipo de Dominio
    type_scores = domains_df.groupby('Domain_Type')['SEO_Score'].mean().nlargest(5).sort_values(ascending=True)
    summary_fig.add_trace(
        go.Bar(
            x=type_scores.values, y=type_scores.index, orientation='h',
            marker_color=px.colors.qualitative.Pastel[3], text=type_scores.round(1), textposition='outside',
        ), row=2, col=1
    )
    
    # 4. Score SEO por Rango de DR
    bins = [0, 30, 50, 70, 100]
    labels = ['Bajo (0-29)', 'Medio (30-49)', 'Alto (50-69)', 'Élite (70-100)']
    domains_df['DR_Group'] = pd.cut(domains_df['DR'], bins=bins, labels=labels, include_lowest=True)
    dr_group_scores = domains_df.groupby('DR_Group')['SEO_Score'].mean().reindex(labels)
    
    summary_fig.add_trace(
        go.Bar(
            x=dr_group_scores.index, y=dr_group_scores.values,
            marker_color=px.colors.sequential.Teal, text=dr_group_scores.round(1), textposition='auto',
        ), row=2, col=2
    )
    
    summary_fig.update_layout(
        height=800, showlegend=False,
        title_text="📝 RESUMEN EJECUTIVO DEL PERFIL DE BACKLINKS",
        title_x=0.5, title_font_size=24, template='plotly_white'
    )
    summary_fig.update_xaxes(title_text="Score SEO Promedio", row=2, col=1)
    summary_fig.update_yaxes(title_text="Score SEO Promedio", row=2, col=2)
    
    return summary_fig

def create_analysis_dashboard(domains_df, in_content_analysis):
    """Crea dashboard completo de análisis organizado."""
    if domains_df.empty: return go.Figure()
        
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            '1. Distribución de Dominios por Rango de DR',
            '2. Score SEO vs. Autoridad de Dominio',
            '3. Top 15 Dominios por Score SEO',
            '4. Distribución por Tipo de Dominio',
            '5. Distribución por Calidad de Dominio',
            '6. Tráfico (Log) vs. Autoridad (DR)',
            '7. Tasa Dofollow (Top 10 Dominios)',
            '8. Relación Tráfico-Score SEO',
            '9. Tipos de Dominio In-Content' if in_content_analysis else '9. Heatmap de Correlación'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'pie'}, {'type': 'pie'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie' if in_content_analysis else 'heatmap'}]
        ],
        vertical_spacing=0.10, horizontal_spacing=0.15, row_heights=[0.33, 0.33, 0.34]
    )
    
    # 1. Distribución DR
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    domains_df['DR_Range'] = pd.cut(domains_df['DR'], bins=bins, labels=labels, include_lowest=True)
    dr_dist = domains_df['DR_Range'].value_counts().sort_index()
    fig.add_trace(go.Bar(x=dr_dist.index, y=dr_dist.values, marker_color=px.colors.sequential.Blues,
                        text=dr_dist.values, textposition='inside', hovertemplate='<b>Rango DR:</b> %{x}<br><b>Dominios:</b> %{y}<extra></extra>'), row=1, col=1)
    
    # 2. Score SEO vs DR
    fig.add_trace(go.Scatter(x=domains_df['DR'], y=domains_df['SEO_Score'], mode='markers',
                            marker=dict(size=domains_df['Total_Links']/domains_df['Total_Links'].max() * 20 + 5,
                                        color=domains_df['Traffic'].apply(lambda x: np.log10(x+1)), colorscale='Viridis', showscale=True),
                            text=domains_df.apply(lambda row: f"<b>{row['Domain']}</b><br>DR: {row['DR']}<br>Score: {row['SEO_Score']}<br>Tráfico: {row['Traffic']:,.0f}", axis=1),
                            hoverinfo='text'), row=1, col=2)
    
    # 3. Top 15 dominios
    top_domains = domains_df.nlargest(15, 'SEO_Score').sort_values(by='SEO_Score', ascending=True)
    fig.add_trace(go.Bar(y=top_domains['Domain'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x), x=top_domains['SEO_Score'], orientation='h', marker_color='#2E86AB',
                        text=top_domains['SEO_Score'].round(1), textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Score SEO: %{x:.1f}<br>DR: %{customdata[0]}<br>Tráfico: %{customdata[1]:,.0f}<extra></extra>', customdata=top_domains[['DR', 'Traffic']]), row=1, col=3)
    
    # 4. Distribución por tipo
    type_dist = domains_df['Domain_Type'].value_counts()
    fig.add_trace(go.Pie(labels=type_dist.index, values=type_dist.values, hole=0.4, marker_colors=px.colors.qualitative.Set3, textinfo='percent'), row=2, col=1)
    
    # 5. Distribución por calidad
    quality_dist = domains_df['Domain_Quality'].value_counts()
    fig.add_trace(go.Pie(labels=quality_dist.index, values=quality_dist.values, hole=0.3, marker_colors=['#2E8B57', '#FFA500', '#FF6347'], textinfo='percent'), row=2, col=2)
    
    # 6. Tráfico (Log) vs DR
    fig.add_trace(go.Scatter(x=domains_df['DR'], y=domains_df['Traffic'].apply(lambda x: np.log10(x+1)), mode='markers',
                            marker=dict(size=8, color=domains_df['SEO_Score'], colorscale='RdYlGn', showscale=False),
                            text=domains_df.apply(lambda row: f"<b>{row['Domain']}</b><br>DR: {row['DR']}<br>Tráfico: {row['Traffic']:,.0f}", axis=1), hoverinfo='text'), row=2, col=3)
    
    # 7. Tasa Dofollow
    domains_df['Dofollow_Rate'] = ((domains_df['Total_Links'] - domains_df['Nofollow_Count']) / domains_df['Total_Links']) * 100
    top_dofollow = domains_df.nlargest(10, 'Dofollow_Rate').sort_values(by='Dofollow_Rate', ascending=True)
    fig.add_trace(go.Bar(x=top_dofollow['Dofollow_Rate'], y=top_dofollow['Domain'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x),
                        orientation='h', marker_color='#4CAF50', text=top_dofollow['Dofollow_Rate'].round(1).astype(str) + '%', textposition='outside'), row=3, col=1)
    
    # 8. Relación Tráfico-Score
    fig.add_trace(go.Scatter(x=domains_df['Traffic'].apply(lambda x: np.log10(x+1)), y=domains_df['SEO_Score'], mode='markers',
                            marker=dict(size=domains_df['DR']/10 + 5, color=domains_df['DR'], colorscale='Plasma', showscale=False),
                            text=domains_df['Domain'], hoverinfo='text'), row=3, col=2)
    
    # 9. Análisis In-Content o Heatmap
    if in_content_analysis:
        domain_types = in_content_analysis['domain_types']
        fig.add_trace(go.Pie(labels=list(domain_types.keys()), values=list(domain_types.values()), hole=0.5, marker_colors=px.colors.qualitative.Pastel, textinfo='label+percent'), row=3, col=3)
    else:
        numeric_cols = [col for col in domains_df.columns if col in ['DR', 'Traffic', 'Total_Links', 'SEO_Score', 'Nofollow_Count']]
        if len(numeric_cols) > 1:
            corr_matrix = domains_df[numeric_cols].corr()
            fig.add_trace(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0,
                                    text=corr_matrix.round(2).values, texttemplate='%{text}', textfont={"size": 10}, name='Correlación'), row=3, col=3)

    fig.update_layout(height=1300, showlegend=False, title_text="📊 DASHBOARD COMPLETO DE ANÁLISIS DE BACKLINKS", title_x=0.5, title_font_size=24, template='plotly_white')
    fig.update_yaxes(title_text="Score SEO", row=1, col=2)
    fig.update_xaxes(title_text="Score SEO", row=1, col=3)
    fig.update_yaxes(title_text="Log Tráfico (log10)", row=2, col=3)
    fig.update_xaxes(title_text="Tasa Dofollow (%)", row=3, col=1)
    fig.update_xaxes(title_text="Log Tráfico (log10)", row=3, col=2)
    fig.update_yaxes(title_text="Score SEO", row=3, col=2)
    
    return fig

# ==============================================================================
# 5. FUNCIÓN PRINCIPAL DE STREAMLIT
# ==============================================================================

def main():
    # --- Título Principal ---
    st.title("✨ Analizador Completo de Backlinks con IA")
    st.markdown("Herramienta para evaluar el perfil de enlaces, calcular un Score SEO por dominio y generar una estrategia de *link building* usando Gemini AI.")
    st.markdown("---")

    # --- Sidebar para Configuración (PASOS 1-4) ---
    st.sidebar.header("🔑 Configuración de IA")
    api_key = st.sidebar.text_input("Ingresa tu clave de Gemini API", type="password")
    
    client, gemini_available, gen_config, safety_settings = configure_gemini(api_key)
    
    if not api_key:
        st.sidebar.warning("Por favor, ingresa tu clave de API para habilitar el análisis estratégico con IA.")
    elif not gemini_available:
        st.sidebar.error("Error al conectar con la API de Gemini.")
    else:
        st.sidebar.success(f"✅ Gemini ({MODEL_NAME}) configurado.")
    
    st.sidebar.header("⚙️ Parámetros del Análisis")
    
    with st.sidebar.expander("📊 1. Métricas de Referencia del Dominio Objetivo"):
        rd_total = st.number_input("🔗 Dominios de Referencia (RD)", min_value=0, value=1500)
        dr_total = st.slider("🏆 Domain Rating (DR) Objetivo", min_value=0, max_value=100, value=55)
        traffic_total = st.number_input("📈 Tráfico Orgánico Mensual Estimado", min_value=0, value=120000)
        total_backlinks = st.number_input("🔗 Total de Backlinks conocidos (del archivo)", min_value=0, value=10000)
    
    with st.sidebar.expander("🔎 2. Filtros y Muestreo"):
        link_type = st.text_input("Tipo de enlace específico a analizar (Ej: 'guest post' - opcional)", value="").strip()
        target_keywords_input = st.text_input("Palabras clave principales (separadas por comas - opcional)", value="")
        target_keywords = [kw.strip().lower() for kw in target_keywords_input.split(',') if kw.strip()]
        
        sample_size = st.number_input("Número máximo de enlaces a analizar (Muestreo - opcional)", min_value=0, value=0)
        if sample_size == 0: sample_size = None
    
    # --- Carga de Archivo (PASO 5) ---
    st.sidebar.header("📥 Carga de Archivo")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o XLSX de Backlinks (Ahrefs, SEMrush, etc.)", type=["csv", "xlsx"])
    
    if uploaded_file is None:
        st.info("Sube tu archivo de backlinks y presiona 'Iniciar Análisis' para comenzar.")
        return

    # --- Botón de Ejecución ---
    if st.sidebar.button("🚀 Iniciar Análisis Completo", key="run_analysis"):
        
        metrics = {
            "rd_total": rd_total, "dr_total": dr_total, "traffic_total": traffic_total, 
            "total_backlinks": total_backlinks
        }
        
        # 1. Procesamiento de Datos
        df = load_and_filter_data(uploaded_file, link_type, sample_size)
        
        if df.empty:
            st.error("No se pudieron procesar los datos. Por favor, revisa la estructura del archivo y las columnas requeridas.")
            return

        # 2. Cálculo de Métricas Agregadas
        domains_df, metrics_summary = calculate_metrics(df, metrics['total_backlinks'])
        
        if domains_df.empty:
            st.error("No se encontraron dominios válidos después del procesamiento.")
            return

        st.success(f"✅ Dataset procesado: {metrics_summary['total_domains']:,} dominios únicos analizados.")

        # 3. Generación de Visualizaciones
        st.header("📈 Resultados del Análisis")
        st.markdown("---")
        
        summary_fig = create_executive_summary(metrics, domains_df)
        st.plotly_chart(summary_fig, use_container_width=True)
        
        # 4. Análisis y Estrategia IA (en un tab)
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard Detallado", "🧠 Estrategia de IA (In-Content)", "📋 Data Procesada"])
        
        with tab1:
            dashboard_fig = create_analysis_dashboard(domains_df, analyze_in_content_patterns(df))
            st.plotly_chart(dashboard_fig, use_container_width=True)

        with tab2:
            st.subheader(f"Estrategia de Link Building para Enlaces 'In-Content'")
            
            in_content_analysis = analyze_in_content_patterns(df)
            
            if not gemini_available:
                st.warning("El análisis estratégico con IA está deshabilitado. Por favor, ingresa tu clave de API de Gemini.")
            elif not in_content_analysis:
                st.info("No se encontraron suficientes enlaces con patrones 'in-content' para generar una estrategia de IA específica.")
            else:
                strategy_text = generate_in_content_strategy(
                    client, gen_config, safety_settings, in_content_analysis, target_keywords, metrics
                )
                st.markdown(f"**Modelo Utilizado:** `{MODEL_NAME}`")
                st.markdown("---")
                st.markdown(strategy_text)

        with tab3:
            st.subheader("Dataset de Dominios Analizados (Top 100)")
            st.dataframe(domains_df.nlargest(100, 'SEO_Score'), use_container_width=True)


if __name__ == "__main__":
    main()
