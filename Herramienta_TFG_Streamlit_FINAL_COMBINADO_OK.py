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
# Procesamiento Central de Datos
# ==========================================
@st.cache_data
def process_data(_df):
    df_proc = _df.copy()
    required_columns = ['Edad', 'Antigüedad', 'Desempeño', 'Salario', 'Formación_Reciente', 'Clima_Laboral', 'Departamento', 'Riesgo_Abandono', 'Horas_Extra', 'Bajas_Último_Año', 'Promociones_2_Años', 'Tipo_Contrato']
    missing = [col for col in required_columns if col not in df_proc.columns]
    if missing:
        return None, f"Faltan las columnas: {', '.join(missing)}", None, None

    df_encoded = pd.get_dummies(df_proc, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
    
    if 'Riesgo_Abandono' not in df_encoded.columns:
         return None, "La columna 'Riesgo_Abandono' es necesaria.", None, None

    X = df_encoded.drop("Riesgo_Abandono", axis=1)
    y = df_encoded["Riesgo_Abandono"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    df_sim = df_proc.copy()
    df_sim["Prob_Abandono"] = model.predict_proba(X_scaled)[:, 1]

    def generate_detailed_recommendation(row):
        risk, clima, desempeno, antiguedad = row['Prob_Abandono'], row['Clima_Laboral'], row['Desempeño'], row['Antigüedad']
        if risk > 0.75:
            if clima < 3: return "Riesgo CRÍTICO por bajo clima laboral. ACCIÓN URGENTE: Intervenir en el equipo y hablar con el empleado sobre su bienestar."
            if desempeno >= 4: return "Riesgo ALTO en un empleado clave. Posible falta de retos o reconocimiento. ACCIÓN: Revisar plan de carrera y compensación."
            return "Riesgo ALTO. Investigar causas. ACCIÓN: Programar una entrevista de seguimiento."
        elif risk > 0.4:
            if antiguedad < 2: return "Riesgo MEDIO en empleado nuevo. Posible problema de adaptación. ACCIÓN: Reforzar 'onboarding' y asignar un mentor."
            return "Riesgo MEDIO. En observación. ACCIÓN: Fomentar formación y dar feedback para aumentar compromiso."
        else:
            return "Riesgo BAJO. Empleado comprometido. ACCIÓN: Mantener condiciones y ofrecer desarrollo a largo plazo."
            
    df_sim['Recomendación'] = df_sim.apply(generate_detailed_recommendation, axis=1)

    features_cluster = ["Edad", "Antigüedad", "Desempeño", "Salario", "Formación_Reciente", "Clima_Laboral", "Horas_Extra", "Bajas_Último_Año", "Promociones_2_Años"]
    X_cluster = df_sim[features_cluster]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    perfil_dict = {0: "Potencial Crecimiento", 1: "Bajo Compromiso", 2: "Alto Desempeño", 3: "En Riesgo"}
    df_sim["Perfil_Empleado"] = pd.Series(clusters).map(perfil_dict)
    
    return df_sim, None, model, X.columns

# ==========================================
# Cuerpo Principal de la Aplicación
# ==========================================
st.title("🚀 Herramienta IA de Planificación Estratégica de RRHH")

if uploaded_file is None:
    st.info("ℹ️ Para comenzar, sube un archivo CSV usando el menú de la izquierda.")
    st.stop()

df_original = pd.read_csv(uploaded_file, sep=";")
df_sim, error_message, model, feature_names = process_data(df_original.copy())

if error_message:
    st.error(f"❌ {error_message}")
    st.stop()

# --- Filtros ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Filtros del Informe")
dept_list = ['Todos'] + sorted(df_sim['Departamento'].unique().tolist())
perfil_list = ['Todos'] + sorted(df_sim['Perfil_Empleado'].unique().tolist())

dept_selection = st.sidebar.selectbox('Filtrar por Departamento', options=dept_list)
perfil_selection = st.sidebar.selectbox('Filtrar por Perfil', options=perfil_list)

df_filtered = df_sim.copy()
if dept_selection != 'Todos':
    df_filtered = df_filtered[df_filtered['Departamento'] == dept_selection]
if perfil_selection != 'Todos':
    df_filtered = df_filtered[df_filtered['Perfil_Empleado'] == perfil_selection]

# ==========================================
# Nueva Estructura de Pestañas
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📈 Dashboard y Resumen Ejecutivo", 
    "👥 Análisis por Segmentos", 
    "🧑‍💻 Consulta de Empleados y Simulación"
])

# --- PESTAÑA 1: DASHBOARD Y RESUMEN EJECUTIVO ---
with tab1:
    st.header("Dashboard y Resumen Ejecutivo")
    
    filter_text = "toda la plantilla"
    if dept_selection != 'Todos' or perfil_selection != 'Todos':
        filter_text = "la selección filtrada"

    st.markdown(f"A continuación se muestran los indicadores y conclusiones clave para **{filter_text}**.")
    
    if df_filtered.empty:
        st.warning("La selección de filtros no ha devuelto ningún empleado.")
    else:
        # --- KPIs ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("👥 Empleados", f"{len(df_filtered)}")
        kpi2.metric("🔥 Riesgo Medio", f"{df_filtered['Prob_Abandono'].mean():.1%}")
        kpi3.metric("😊 Clima Medio", f"{df_filtered['Clima_Laboral'].mean():.2f}/5")
        kpi4.metric("💰 Salario Medio", f"€{df_filtered['Salario'].mean():,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            # --- Conclusiones Estratégicas ---
            st.subheader("🎯 Conclusiones y Estrategias Clave")
            st.success("**Identificar y retener talento joven:** Especialmente aquellos con alto potencial y riesgo de fuga. [cite: 37]")
            st.warning("**Fomentar la formación:** Priorizar áreas con baja motivación o desempeño para mejorar el compromiso. [cite: 38]")
            st.info("**Implementar medidas personalizadas:** Usar la segmentación por perfiles para adaptar las acciones de RRHH. [cite: 38]")
            st.error("**Actuar sobre el clima laboral:** Promover entrevistas de seguimiento en departamentos con bajo clima para atajar problemas de raíz. [cite: 39]")

        with col2:
            # --- Top 5 Empleados en Riesgo ---
            st.subheader("Top 5 Empleados con Mayor Riesgo")
            top_5_risk = df_filtered.nlargest(5, 'Prob_Abandono')
            for index, row in top_5_risk.iterrows():
                riesgo_color = "red" if row.get('Prob_Abandono', 0) >= 0.75 else "orange"
                st.markdown(f"""
                <div style="border-left: 5px solid {riesgo_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #f8f9fa;">
                    **ID {index}** ({row['Departamento']}) - **Riesgo: {row['Prob_Abandono']:.1%}** <br>
                    <small>{row['Recomendación']}</small>
                </div>
                """, unsafe_allow_html=True)

# --- PESTAÑA 2: ANÁLISIS POR SEGMENTOS ---
with tab2:
    st.header("Análisis por Segmentos (Perfiles y Departamentos)")
    if df_filtered.empty:
        st.warning("La selección de filtros no ha devuelto ningún empleado.")
    else:
        st.subheader("Análisis de Perfiles de Empleados (Clusters)")
        st.markdown("A continuación se describen los perfiles identificados en el grupo seleccionado.")
        
        # --- Resumen de Perfiles ---
        perfiles_en_filtro = df_filtered['Perfil_Empleado'].unique()
        for perfil in perfiles_en_filtro:
            grupo = df_filtered[df_filtered['Perfil_Empleado'] == perfil]
            with st.expander(f"**Perfil: '{perfil}'** ({len(grupo)} empleados)"):
                st.markdown(f"Este grupo se caracteriza por tener un **riesgo de abandono medio de {grupo['Prob_Abandono'].mean():.1%}** y un **clima laboral de {grupo['Clima_Laboral'].mean():.1f}/5**.")
                st.markdown(f"Su **desempeño medio es de {grupo['Desempeño'].mean():.1f}/5** y su **antigüedad media es de {grupo['Antigüedad'].mean():.1f} años**.")
                st.info(f"**Recomendación principal para este grupo:** {grupo['Recomendación'].mode()[0]}")

        st.markdown("---")
        
        # --- Gráficas por Departamento ---
        st.subheader("Análisis por Departamento")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Clima Laboral Medio")
            clima_dpto = df_filtered.groupby('Departamento')['Clima_Laboral'].mean().sort_values()
            fig, ax = plt.subplots(); clima_dpto.plot(kind='barh', ax=ax, color='c'); st.pyplot(fig)
        with col2:
            st.markdown("##### Riesgo de Abandono Medio")
            riesgo_dpto = df_filtered.groupby('Departamento')['Prob_Abandono'].mean().sort_values()
            fig, ax = plt.subplots(); riesgo_dpto.plot(kind='barh', ax=ax, color='salmon'); st.pyplot(fig)

# --- PESTAÑA 3: CONSULTA Y SIMULACIÓN ---
with tab3:
    st.header("Consulta de Empleados y Simulación de Políticas")
    if df_filtered.empty:
        st.warning("La selección de filtros no ha devuelto ningún empleado.")
    else:
        # --- Tabla Completa ---
        st.subheader("Tabla de Empleados (Filtrada)")
        st.markdown("Aquí puedes ver y ordenar la lista completa de empleados que coinciden con tu selección.")
        st.dataframe(df_filtered[['Departamento', 'Perfil_Empleado', 'Edad', 'Antigüedad', 'Desempeño', 'Clima_Laboral', 'Prob_Abandono', 'Recomendación']])
        
        st.markdown("---")
        
        # --- Simulador Interactivo ---
        st.subheader("🕹️ Simulador de Políticas 'What-If'")
        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            st.markdown("Ajusta el impacto esperado de cada política para ver cómo afectaría al riesgo del grupo de empleados mostrado en la tabla.")
            form_impact = st.slider("Impacto por Mejora de Formación (%)", 0, 50, 10, key="sim_form_tab3")
            sal_impact = st.slider("Impacto por Mejora Salarial (%)", 0, 50, 15, key="sim_sal_tab3")
        
        with sim_col2:
            form_sim = df_filtered['Prob_Abandono'].mean() * (1 - form_impact / 100)
            sal_sim = df_filtered['Prob_Abandono'].mean() * (1 - sal_impact / 100)
            both_sim = df_filtered['Prob_Abandono'].mean() * (1 - (form_impact + sal_impact) / 100)
            escenarios_sim = {'Estado Actual': df_filtered["Prob_Abandono"].mean(), 'Mejora Formación': form_sim, 'Mejora Salarial': sal_sim, 'Política Combinada': both_sim}
            
            fig_sim, ax_sim = plt.subplots()
            bars = sns.barplot(x=list(escenarios_sim.keys()), y=list(escenarios_sim.values()), palette="viridis", ax=ax_sim)
            for bar in bars.patches:
                ax_sim.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1%}', ha='center', va='bottom', fontweight='bold')
            ax_sim.set_ylabel("Riesgo Medio de Abandono")
            st.pyplot(fig_sim)
