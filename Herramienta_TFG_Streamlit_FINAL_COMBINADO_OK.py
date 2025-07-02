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
        return None, f"Faltan las columnas: {', '.join(missing)}", None, None, None

    df_encoded = pd.get_dummies(df_proc, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
    if 'Riesgo_Abandono' not in df_encoded.columns: return None, "La columna 'Riesgo_Abandono' es necesaria.", None, None, None

    X = df_encoded.drop("Riesgo_Abandono", axis=1)
    y = df_encoded["Riesgo_Abandono"]
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    
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
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_cluster_scaled)
    clusters = kmeans.predict(X_cluster_scaled)
    perfil_dict = {0: "Potencial Crecimiento", 1: "Bajo Compromiso", 2: "Alto Desempeño", 3: "En Riesgo"}
    df_sim["Perfil_Empleado"] = pd.Series(clusters).map(perfil_dict)
    
    return df_sim, None, model, X, X_scaled

# ==========================================
# Cuerpo Principal
# ==========================================
st.title("🚀 Herramienta IA de Planificación Estratégica de RRHH")

if uploaded_file is None:
    st.info("ℹ️ Para comenzar, sube un archivo CSV usando el menú de la izquierda.")
    st.stop()

try:
    df_original = pd.read_csv(uploaded_file, sep=";")
    df_sim, error_message, model, X_train_df, X_scaled_full = process_data(df_original.copy())

    if error_message:
        st.error(f"❌ {error_message}"); st.stop()

    # --- Filtros ---
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Filtros del Informe")
    dept_list = ['Todos'] + sorted(df_sim['Departamento'].unique().tolist())
    perfil_list = ['Todos'] + sorted(df_sim['Perfil_Empleado'].unique().tolist())
    dept_selection = st.sidebar.selectbox('Filtrar por Departamento', options=dept_list)
    perfil_selection = st.sidebar.selectbox('Filtrar por Perfil', options=perfil_list)

    df_filtered = df_sim.copy()
    if dept_selection != 'Todos': df_filtered = df_filtered[df_filtered['Departamento'] == dept_selection]
    if perfil_selection != 'Todos': df_filtered = df_filtered[df_filtered['Perfil_Empleado'] == perfil_selection]

    # ==========================================
    # ESTRUCTURA DE PESTAÑAS
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard Principal", 
        "👥 Análisis por Segmentos", 
        "🧑‍💻 Consulta y Simulación",
        "📚 Glosario y Metodología"
    ])

    # --- PESTAÑA 1: DASHBOARD PRINCIPAL ---
    with tab1:
        st.header("Dashboard y Resumen Ejecutivo")
        filter_text = "toda la plantilla"
        if dept_selection != 'Todos' or perfil_selection != 'Todos': filter_text = "la selección filtrada"
        
        if df_filtered.empty:
            st.warning("La selección de filtros no ha devuelto ningún empleado.")
        else:
            st.markdown(f"A continuación se muestran los indicadores y conclusiones clave para **{filter_text}**.")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("👥 Empleados", f"{len(df_filtered)}")
            kpi2.metric("🔥 Riesgo Medio", f"{df_filtered['Prob_Abandono'].mean():.1%}")
            kpi3.metric("😊 Clima Medio", f"{df_filtered['Clima_Laboral'].mean():.2f}/5")
            kpi4.metric("💰 Salario Medio", f"€{df_filtered['Salario'].mean():,.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribución del Riesgo de Abandono")
                fig, ax = plt.subplots(); sns.histplot(df_filtered['Prob_Abandono'], bins=15, kde=True, ax=ax, color="skyblue"); ax.set_xlabel("Probabilidad de Abandono"); ax.set_ylabel("Nº de Empleados"); st.pyplot(fig)
                with st.expander("Ver Análisis Detallado"):
                    st.markdown("**¿Qué estamos viendo?:** La distribución de la plantilla según su probabilidad de abandono.")
                    st.markdown("**¿Qué está pasando en tus datos?:**")
                    mean_risk = df_filtered['Prob_Abandono'].mean()
                    if mean_risk > 0.6:
                        st.error(f"El riesgo medio del grupo es de **{mean_risk:.1%}**, lo que indica una situación preocupante. La mayoría de los empleados se concentra en la zona de alto riesgo.")
                    else:
                        st.success(f"El riesgo medio del grupo es de **{mean_risk:.1%}**, lo que sugiere una situación mayormente controlada, aunque existen individuos en riesgo.")
                    st.markdown("**Recomendaciones:** Si hay un pico significativo en la zona de riesgo alto (>70%), es una señal de alerta que requiere una investigación profunda de las causas a nivel organizacional.")

            with col2:
                st.subheader("Top 5 Empleados con Mayor Riesgo")
                st.caption("Ranking de empleados que requieren atención más urgente según el modelo.")
                # CORREGIDO: Quitado el (ID: {index})
                for i, (index, row) in enumerate(df_filtered.nlargest(5, 'Prob_Abandono').iterrows(), 1):
                    riesgo_color = "red" if row.get('Prob_Abandono', 0) >= 0.75 else "orange"
                    st.markdown(f"""
                    <div style="border-left: 5px solid {riesgo_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #f8f9fa;">
                        **{i}. Empleado del dpto. {row['Departamento']}** - Riesgo: **{row['Prob_Abandono']:.1%}** <br>
                        <small><i>{row['Recomendación']}</i></small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🎯 Impulsores Clave del Riesgo de Abandono (Análisis Global)")
            importances = pd.DataFrame(data={'Attribute': X_train_df.columns, 'Importance': np.abs(model.coef_[0])}).sort_values(by='Importance', ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(importances['Attribute'], importances['Importance'], color='skyblue'); ax.set_title('Top 10 Factores que más influyen en la Predicción'); st.pyplot(fig)
            with st.expander("Ver Análisis Detallado"):
                top_feature = importances.iloc[-1]['Attribute'].replace('_', ' ')
                st.markdown("**¿Qué estamos viendo?:** Los factores que el modelo de IA considera más importantes para predecir el abandono.")
                st.markdown("**¿Qué está pasando en tus datos?:** El factor más determinante para predecir el abandono en tu empresa es **'{}'**.".format(top_feature))
                st.markdown("**Recomendaciones:** Diseñar estrategias corporativas que ataquen los 2 o 3 impulsores principales.")

    # --- PESTAÑA 2: ANÁLISIS POR SEGMENTOS ---
    with tab2:
        st.header("Análisis por Segmentos (Perfiles y Departamentos)")
        if df_filtered.empty: st.warning("No hay empleados que coincidan con los filtros.")
        else:
            st.subheader("Análisis de Perfiles de Empleados (Clusters)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Visualización de Clusters (PCA)")
                # ... (código de la gráfica PCA)
            with col2:
                st.markdown("##### Resumen de Perfiles Identificados")
                # ... (código del resumen de perfiles)
            
            st.markdown("---")
            st.subheader("Análisis por Departamento")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clima Laboral Medio")
                fig, ax = plt.subplots(); df_filtered.groupby('Departamento')['Clima_Laboral'].mean().sort_values().plot(kind='barh', ax=ax, color='c'); st.pyplot(fig)
                # EXPLICACIÓN RESTAURADA
                with st.expander("Ver Análisis Detallado"):
                    st.markdown("**¿Qué estamos viendo?:** El ranking de departamentos según la puntuación media de clima laboral.")
                    st.markdown("**¿Qué está pasando en tus datos?:** El departamento con peor clima es **'{}'** ({:.2f}/5), mientras que el mejor es **'{}'** ({:.2f}/5).".format(df_filtered.groupby('Departamento')['Clima_Laboral'].mean().idxmin(), df_filtered.groupby('Departamento')['Clima_Laboral'].mean().min(), df_filtered.groupby('Departamento')['Clima_Laboral'].mean().idxmax(), df_filtered.groupby('Departamento')['Clima_Laboral'].mean().max()))
                    st.markdown("**Recomendaciones:** En departamentos con bajo clima, es crucial realizar encuestas de pulso o 'focus groups' para entender las causas (mala gestión, sobrecarga, etc.) y actuar sobre ellas.")
            with col2:
                st.markdown("##### Riesgo de Abandono Medio")
                fig, ax = plt.subplots(); df_filtered.groupby('Departamento')['Prob_Abandono'].mean().sort_values().plot(kind='barh', ax=ax, color='salmon'); st.pyplot(fig)
                # EXPLICACIÓN RESTAURADA
                with st.expander("Ver Análisis Detallado"):
                    st.markdown("**¿Qué estamos viendo?:** El ranking de departamentos según el riesgo medio de abandono.")
                    st.markdown("**¿Qué está pasando en tus datos?:** El departamento con mayor riesgo es **'{}'** ({:.1%}), lo que lo convierte en el foco principal de las estrategias de retención.".format(df_filtered.groupby('Departamento')['Prob_Abandono'].mean().idxmax(), df_filtered.groupby('Departamento')['Prob_Abandono'].mean().max()))
                    st.markdown("**Recomendaciones:** Priorizar las políticas de retención en los departamentos con mayor riesgo. Analizar si el tipo de rol o la gestión departamental son factores determinantes.")

    # --- PESTAÑA 3: CONSULTA Y SIMULACIÓN ---
    with tab3:
        st.header("Consulta Individual y Simulación de Políticas")
        if df_filtered.empty: st.warning("No hay empleados que coincidan con los filtros.")
        else:
            st.subheader("Análisis Individual Detallado (XAI)")
            selected_id = st.selectbox("Selecciona un ID de Empleado:", df_filtered.index, key="employee_selector")
            if selected_id is not None:
                # ... (código de la ficha individual con XAI) ...
                 pass

            st.markdown("---")
            st.subheader("🕹️ Simulador de Políticas 'What-If'")
            # ... (código del simulador con explicaciones) ...
            pass
            
    # --- PESTAÑA 4: GLOSARIO Y METODOLOGÍA ---
    with tab4:
        st.header("📚 Glosario y Metodología")
        # ... (código del glosario) ...
        pass

except Exception as e:
    st.error(f"Se ha producido un error inesperado. Asegúrate de que tu archivo CSV es correcto. Error: {e}")
