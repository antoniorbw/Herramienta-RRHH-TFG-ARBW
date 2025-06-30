# ==========================================
# Librerías y Configuración Inicial
# ==========================================
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import os
import io

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Herramienta IA Estratégica - RRHH", layout="wide")

# ==========================================
# Barra Lateral (Sidebar)
# ==========================================
st.sidebar.title("⚙️ Panel de Control")

# --- Descarga de plantilla CSV ---
@st.cache_data
def create_template_csv():
    template_data = {
        'Edad': [35, 42, 28, 50, 31, 25, 38, 45, 29, 33], 'Antigüedad': [5, 10, 2, 20, 3, 1, 8, 15, 4, 6],
        'Desempeño': [3, 4, 5, 4, 2, 4, 3, 5, 2, 4], 'Salario': [35000, 55000, 60000, 75000, 32000, 40000, 48000, 85000, 33000, 45000],
        'Formación_Reciente': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1], 'Clima_Laboral': [3, 4, 5, 2, 1, 4, 3, 5, 2, 4],
        'Departamento': ['Ventas', 'TI', 'Marketing', 'Ventas', 'TI', 'Marketing', 'Producción', 'Producción', 'RRHH', 'RRHH'],
        'Riesgo_Abandono': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0], 'Horas_Extra': [5, 2, 0, 8, 10, 1, 4, 0, 6, 2],
        'Bajas_Último_Año': [1, 0, 0, 2, 3, 0, 1, 0, 2, 0], 'Promociones_2_Años': [0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
        'Tipo_Contrato': ['Indefinido', 'Indefinido', 'Temporal', 'Indefinido', 'Temporal', 'Indefinido', 'Indefinido', 'Indefinido', 'Temporal', 'Indefinido']
    }
    df_template = pd.DataFrame(template_data)
    return df_template.to_csv(index=False, sep=';').encode('utf-8')

st.sidebar.download_button(
   label="📥 Descargar plantilla de ejemplo",
   data=create_template_csv(),
   file_name='plantilla_datos_empleados.csv',
   mime='text/csv',
)

# --- Carga de datos ---
uploaded_file = st.sidebar.file_uploader("📤 Sube tu archivo CSV aquí", type=["csv"])

# ==========================================
# Cuerpo Principal de la Aplicación
# ==========================================
st.title("🚀 Herramienta IA de Planificación Estratégica de RRHH")

if uploaded_file is None:
    st.info("ℹ️ Para comenzar, sube un archivo CSV usando el menú de la izquierda. Puedes descargar la plantilla de ejemplo para ver el formato requerido.")
    st.stop()

# --- Procesamiento y Modelado (se cachea para no re-ejecutar innecesariamente) ---
@st.cache_data
def process_data(df):
    required_columns = ['Edad', 'Antigüedad', 'Desempeño', 'Salario', 'Formación_Reciente',
                        'Clima_Laboral', 'Departamento', 'Riesgo_Abandono', 'Horas_Extra', 
                        'Bajas_Último_Año', 'Promociones_2_Años', 'Tipo_Contrato']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return None, f"El archivo no es válido. Faltan las columnas: {', '.join(missing)}"

    df_encoded = pd.get_dummies(df, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
    X = df_encoded.drop("Riesgo_Abandono", axis=1)
    y = df_encoded["Riesgo_Abandono"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    prob_abandono_original = model.predict_proba(X_scaled)[:, 1]
    df_sim = df.copy()
    df_sim["Prob_Abandono"] = prob_abandono_original

    def generate_detailed_recommendation(row):
        risk, clima, desempeno, antiguedad = row['Prob_Abandono'], row['Clima_Laboral'], row['Desempeño'], row['Antigüedad']
        if risk > 0.75:
            if clima < 3: return "Riesgo CRÍTICO. El bajo clima laboral es un factor clave. ACCIÓN URGENTE: Intervenir en el equipo y hablar con el empleado sobre su bienestar."
            if desempeno >= 4: return "Riesgo ALTO en un empleado de alto desempeño. Posible falta de retos o reconocimiento. ACCIÓN: Revisar plan de carrera y compensación."
            return "Riesgo ALTO. Investigar causas específicas. ACCIÓN: Programar una entrevista de seguimiento para entender sus motivaciones y preocupaciones."
        elif risk > 0.4:
            if antiguedad < 2: return "Riesgo MEDIO en empleado nuevo. Posible problema de adaptación. ACCIÓN: Reforzar el proceso de 'onboarding' y asignar un mentor."
            return "Riesgo MEDIO. Empleado en zona de observación. ACCIÓN: Fomentar la formación y ofrecer feedback constructivo para aumentar su compromiso."
        else:
            return "Riesgo BAJO. Empleado comprometido. ACCIÓN: Mantener buenas condiciones y ofrecer oportunidades de desarrollo a largo plazo."
    df_sim['Recomendación'] = df_sim.apply(generate_detailed_recommendation, axis=1)

    features = ["Edad", "Antigüedad", "Desempeño", "Salario", "Formación_Reciente", "Clima_Laboral", "Horas_Extra", "Bajas_Último_Año", "Promociones_2_Años"]
    X_cluster = df_sim[features]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    perfil_dict = {0: "Potencial Crecimiento", 1: "Bajo Compromiso", 2: "Alto Desempeño", 3: "En Riesgo"}
    df_sim["Perfil_Empleado"] = pd.Series(clusters).map(perfil_dict)
    
    return df_sim, None

df_original = pd.read_csv(uploaded_file, sep=";")
df_sim, error_message = process_data(df_original)

if error_message:
    st.error(f"❌ {error_message}")
    st.stop()

st.success(f"✅ Archivo **{uploaded_file.name}** cargado y procesado. Se han analizado **{len(df_sim)}** empleados.")

# ==========================================
# Filtros Interactivos en la Barra Lateral
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("📊 Filtros del Informe")
dept_selection = st.sidebar.multiselect('Filtrar por Departamento', options=df_sim['Departamento'].unique(), default=[])
perfil_selection = st.sidebar.multiselect('Filtrar por Perfil', options=df_sim['Perfil_Empleado'].unique(), default=[])

df_filtered = df_sim.copy()
if dept_selection:
    df_filtered = df_filtered[df_filtered['Departamento'].isin(dept_selection)]
if perfil_selection:
    df_filtered = df_filtered[df_filtered['Perfil_Empleado'].isin(perfil_selection)]

# ==========================================
# 1. Dashboard Estratégico y KPIs
# ==========================================
st.markdown("---")
st.header("📈 Dashboard Estratégico")

# --- Texto dinámico para el dashboard ---
filter_text = "toda la plantilla"
if dept_selection and perfil_selection:
    filter_text = f"los perfiles '{', '.join(perfil_selection)}' en los departamentos '{', '.join(dept_selection)}'"
elif dept_selection:
    filter_text = f"el/los departamento(s) '{', '.join(dept_selection)}'"
elif perfil_selection:
    filter_text = f"el/los perfile(s) '{', '.join(perfil_selection)}'"
st.markdown(f"A continuación se muestran los indicadores clave para **{filter_text}**.")

if df_filtered.empty:
    st.warning("La selección de filtros no ha devuelto ningún empleado. Por favor, ajusta los filtros.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Empleados Analizados", f"{len(df_filtered)}")
col2.metric("🔥 Riesgo Medio de Abandono", f"{df_filtered['Prob_Abandono'].mean():.1%}")
col3.metric("😊 Clima Laboral Medio", f"{df_filtered['Clima_Laboral'].mean():.2f}/5")
col4.metric("💰 Salario Medio", f"€{df_filtered['Salario'].mean():,.0f}")

# ==========================================
# 2. Simulador Interactivo de Políticas
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Simulador de Políticas")
st.sidebar.caption("Ajusta el impacto estimado de cada política para ver los resultados en tiempo real.")
form_impact = st.sidebar.slider("Impacto Mejora Formación (%)", 0, 50, 10)
sal_impact = st.sidebar.slider("Impacto Mejora Salarial (%)", 0, 50, 15)

with st.expander("▶️ Ver Simulación de Políticas Estratégicas"):
    form_sim = df_filtered['Prob_Abandono'].mean() * (1 - form_impact / 100)
    sal_sim = df_filtered['Prob_Abandono'].mean() * (1 - sal_impact / 100)
    both_sim = df_filtered['Prob_Abandono'].mean() * (1 - (form_impact + sal_impact) / 100)
    
    escenarios_sim = {
        'Estado Actual': df_filtered["Prob_Abandono"].mean(),
        'Mejora Formación': form_sim,
        'Mejora Salarial': sal_sim,
        'Política Combinada': both_sim
    }

    fig_sim, ax_sim = plt.subplots()
    bars = sns.barplot(x=list(escenarios_sim.keys()), y=list(escenarios_sim.values()), palette="viridis", ax=ax_sim)
    ax_sim.set_title("Impacto Estimado de Políticas en el Riesgo Medio")
    ax_sim.set_ylabel("Probabilidad Media de Abandono")
    for bar in bars.patches:
        ax_sim.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1%}', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig_sim)
    
    st.caption(f"🔍 **Interpretación:** Aplicando los impactos seleccionados a **{filter_text}**, la política más efectiva sería la **Combinada**, reduciendo el riesgo medio de abandono del **{escenarios_sim['Estado Actual']:.1%}** al **{escenarios_sim['Política Combinada']:.1%}**.")

# ==========================================
# 3. Análisis Profundo de Perfiles
# ==========================================
st.markdown("---")
st.header("👤 Análisis Profundo de Perfiles")

# --- Normalización de datos para la gráfica de radar ---
radar_features = ['Desempeño', 'Clima_Laboral', 'Salario', 'Antigüedad', 'Horas_Extra']
scaler_radar = MinMaxScaler()
df_radar = df_sim.copy()
df_radar[radar_features] = scaler_radar.fit_transform(df_radar[radar_features])

# --- Medias por perfil y totales ---
profile_means = df_radar.groupby('Perfil_Empleado')[radar_features].mean()
total_means = df_radar[radar_features].mean()

col1, col2 = st.columns([1.5, 2])
with col1:
    st.markdown("##### Comparativa de Perfiles (Gráfica de Radar)")
    st.caption("Selecciona un perfil para compararlo con la media de la empresa.")
    profile_to_show = st.selectbox("Selecciona un Perfil", options=profile_means.index)
    
    if profile_to_show:
        fig = go.Figure()
        # Radar del perfil seleccionado
        fig.add_trace(go.Scatterpolar(
            r=profile_means.loc[profile_to_show].values,
            theta=radar_features,
            fill='toself',
            name=f'Perfil: {profile_to_show}'
        ))
        # Radar de la media de la empresa
        fig.add_trace(go.Scatterpolar(
            r=total_means.values,
            theta=radar_features,
            fill='toself',
            name='Media de la Empresa',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='orange')
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("##### Ficha del Perfil Seleccionado")
    if profile_to_show:
        grupo = df_sim[df_sim["Perfil_Empleado"] == profile_to_show]
        st.subheader(f"Perfil: '{profile_to_show}'")
        st.metric("Número de Empleados", f"{len(grupo)}")
        
        c1, c2 = st.columns(2)
        c1.metric("Riesgo Medio de Abandono", f"{grupo['Prob_Abandono'].mean():.1%}")
        c2.metric("Salario Medio", f"€{grupo['Salario'].mean():,.0f}")
        
        st.markdown(f"**Recomendación estratégica principal para este grupo:**")
        st.info(f"{grupo['Recomendación'].mode()[0]}")
    
    st.caption("🔍 **Interpretación:** La gráfica de radar permite identificar visualmente las características que definen a cada perfil. Compara el polígono azul (perfil) con el naranja (media de la empresa). Un pico en 'Salario' significa que ese perfil gana más que la media; un pico bajo en 'Clima_Laboral' indica que están menos satisfechos que el promedio.")
