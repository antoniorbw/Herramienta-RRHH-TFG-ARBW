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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fpdf import FPDF
import os
import io

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Herramienta IA - RRHH", layout="wide")

# ==========================================
# Barra Lateral (Sidebar)
# ==========================================
st.sidebar.header("⚙️ Configuración")

# --- Descarga de plantilla CSV ---
@st.cache_data
def create_template_csv():
    template_data = {
        'Edad': [35, 42, 28, 50, 31], 'Antigüedad': [5, 10, 2, 20, 3],
        'Desempeño': [3, 4, 5, 4, 2], 'Salario': [35000, 55000, 60000, 75000, 32000],
        'Formación_Reciente': [1, 0, 1, 0, 1], 'Clima_Laboral': [3, 4, 5, 2, 1],
        'Departamento': ['Ventas', 'TI', 'Marketing', 'Ventas', 'TI'],
        'Riesgo_Abandono': [0, 1, 0, 1, 1], 'Horas_Extra': [5, 2, 0, 8, 10],
        'Bajas_Último_Año': [1, 0, 0, 2, 3], 'Promociones_2_Años': [0, 1, 1, 0, 0],
        'Tipo_Contrato': ['Indefinido', 'Indefinido', 'Temporal', 'Indefinido', 'Temporal']
    }
    df_template = pd.DataFrame(template_data)
    return df_template.to_csv(index=False, sep=';').encode('utf-8')

csv_template = create_template_csv()
st.sidebar.download_button(
   label="📥 Descargar plantilla de ejemplo (.csv)",
   data=csv_template,
   file_name='plantilla_datos_empleados.csv',
   mime='text/csv',
)
st.sidebar.caption("Nota: Si al abrir en Excel los datos no se separan en columnas, utiliza la opción 'Datos' -> 'Desde texto/CSV' y elige 'Punto y coma' como delimitador.")

# --- Carga de datos ---
st.sidebar.header("📤 Carga de Datos del Usuario")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

# ==========================================
# Cuerpo Principal de la Aplicación
# ==========================================
st.title("🧠 Herramienta de IA para la Planificación Estratégica de la Plantilla")
st.markdown(
    """
    **Aplicación desarrollada como parte de un Trabajo de Fin de Grado (TFG)** | Fecha: 24-06-2025

    Esta es una herramienta interactiva que te permite analizar tu plantilla para predecir el riesgo de abandono, 
    identificar perfiles de empleados y simular el impacto de políticas de RRHH.
    """
)

if uploaded_file is None:
    st.info("ℹ️ Para comenzar, sube un archivo CSV con los datos de tus empleados usando el menú de la izquierda. Si no tienes uno, puedes descargar la plantilla de ejemplo.")
    st.stop()

# --- Procesamiento y Modelado (ocurre una vez cargado el archivo) ---
try:
    df = pd.read_csv(uploaded_file, sep=";")
    
    required_columns = ['Edad', 'Antigüedad', 'Desempeño', 'Salario', 'Formación_Reciente',
                        'Clima_Laboral', 'Departamento', 'Riesgo_Abandono', 'Horas_Extra', 
                        'Bajas_Último_Año', 'Promociones_2_Años', 'Tipo_Contrato']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"❌ El archivo no es válido. Faltan las siguientes columnas: {', '.join(missing)}")
        st.stop()

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

    def generate_recommendation(prob):
        if prob >= 0.6: return "Alto riesgo: Considerar programas de retención y revisión salarial/condiciones."
        elif prob >= 0.3: return "Riesgo medio: Monitorear, ofrecer formación o mejoras en clima laboral."
        else: return "Bajo riesgo: Mantener condiciones, enfoque en desarrollo profesional."
    df_sim['Recomendación'] = df_sim['Prob_Abandono'].apply(generate_recommendation)

    features = ["Edad", "Antigüedad", "Desempeño", "Salario", "Formación_Reciente", "Clima_Laboral", 
                "Horas_Extra", "Bajas_Último_Año", "Promociones_2_Años"]
    X_cluster = df_sim[features]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    perfil_dict = {0: "Potencial Crecimiento", 1: "Bajo Compromiso", 2: "Alto Desempeño", 3: "En Riesgo"}
    df_sim["Perfil_Empleado"] = pd.Series(clusters).map(perfil_dict)

except Exception as e:
    st.error(f"Ha ocurrido un error al procesar el archivo. Asegúrate de que usa ';' como separador. Error: {e}")
    st.stop()

st.success(f"✅ Archivo **{uploaded_file.name}** cargado y procesado correctamente. Se han analizado **{len(df)}** empleados.")
st.markdown("---")

# ==============================================================================
# INICIO DEL INFORME INTEGRADO EN LA INTERFAZ
# ==============================================================================

st.header("1. Análisis General de Riesgo de Abandono")
high_risk = (df_sim["Prob_Abandono"] >= 0.6).sum()
medium_risk = ((df_sim["Prob_Abandono"] >= 0.3) & (df_sim["Prob_Abandono"] < 0.6)).sum()
low_risk = (df_sim["Prob_Abandono"] < 0.3).sum()

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("##### Resumen de Niveles de Riesgo")
    st.metric("🔴 Empleados con Riesgo Alto (>60%)", f"{high_risk} empleados")
    st.metric("🟡 Empleados con Riesgo Medio (30-60%)", f"{medium_risk} empleados")
    st.metric("🟢 Empleados con Riesgo Bajo (<30%)", f"{low_risk} empleados")

with col2:
    st.markdown("##### Distribución del Riesgo en la Plantilla")
    fig, ax = plt.subplots()
    sns.histplot(df_sim['Prob_Abandono'], bins=20, kde=True, ax=ax, color="skyblue")
    ax.set_title("Distribución de la Probabilidad de Abandono")
    ax.set_xlabel("Probabilidad de Abandono")
    ax.set_ylabel("Nº de Empleados")
    st.pyplot(fig)
st.caption("🔍 **Interpretación:** Esta gráfica muestra cuántos empleados se encuentran en cada nivel de riesgo. Un pico a la derecha (cerca de 1.0) indica una alta concentración de empleados con riesgo de irse, mientras que un pico a la izquierda (cerca de 0.0) es un signo de una plantilla estable.")

st.header("2. Simulación de Políticas Estratégicas de RRHH")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Se ha simulado el impacto de diferentes políticas sobre el **riesgo medio de abandono** de toda la plantilla para evaluar su efectividad. Las políticas simuladas son:")
    st.markdown("- **Mejora Formación**: Aumento de las oportunidades de desarrollo.")
    st.markdown("- **Mejora Salarial**: Incremento salarial general.")
    st.markdown("- **Política Combinada**: Ambas medidas aplicadas conjuntamente.")

with col2:
    form_sim = df_sim.copy(); form_sim["Prob_Abandono"] *= 0.9
    sal_sim = df_sim.copy(); sal_sim["Prob_Abandono"] *= 0.85
    both_sim = df_sim.copy(); both_sim["Prob_Abandono"] *= 0.8
    escenarios_sim = {
        'Estado Actual': df_sim["Prob_Abandono"].mean(),
        'Mejora Formación': form_sim["Prob_Abandono"].mean(),
        'Mejora Salarial': sal_sim["Prob_Abandono"].mean(),
        'Política Combinada': both_sim["Prob_Abandono"].mean()
    }

    fig_sim, ax_sim = plt.subplots()
    sns.barplot(x=list(escenarios_sim.keys()), y=list(escenarios_sim.values()), palette="viridis", ax=ax_sim)
    ax_sim.set_title("Impacto Estimado de Políticas en el Riesgo Medio")
    ax_sim.set_ylabel("Probabilidad Media de Abandono")
    for index, value in enumerate(escenarios_sim.values()):
        ax_sim.text(index, value, f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig_sim)
st.caption("🔍 **Interpretación:** La barra más baja representa la política más efectiva para reducir el riesgo de abandono a nivel global. Esta simulación ayuda a priorizar las inversiones en RRHH.")

st.header("3. Análisis por Departamento")
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Departamentos con Más Riesgo")
    riesgo_dpto = df_sim.groupby('Departamento')['Prob_Abandono'].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    riesgo_dpto.plot(kind='barh', ax=ax, color='salmon')
    ax.set_title("Riesgo de Abandono Medio por Departamento")
    ax.set_xlabel("Probabilidad Media de Abandono")
    st.pyplot(fig)
    st.caption("🔍 **Interpretación:** Esta gráfica ordena los departamentos del más al menos propenso a la rotación. Es útil para identificar dónde se deben centrar los esfuerzos de retención.")

with col2:
    st.markdown("##### Clima Laboral por Departamento")
    clima_dpto = df_sim.groupby('Departamento')['Clima_Laboral'].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    clima_dpto.plot(kind='barh', ax=ax, color='c')
    ax.set_title("Clima Laboral Medio por Departamento")
    ax.set_xlabel("Puntuación Media (sobre 5)")
    st.pyplot(fig)
    st.caption("🔍 **Interpretación:** Compara la satisfacción y el ambiente de trabajo entre departamentos. Un bajo clima laboral suele estar correlacionado con un alto riesgo de abandono.")

st.header("4. Perfiles de Empleados (Clustering)")
col1, col2 = st.columns([2, 1.5])
with col1:
    st.markdown("##### Visualización de Perfiles (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["Perfil"] = df_sim["Perfil_Empleado"]
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2", s=80, ax=ax3)
    ax3.grid(True)
    ax3.set_title("Agrupación de Perfiles de Empleados")
    st.pyplot(fig3)
    st.caption("🔍 **Interpretación:** Cada punto es un empleado. Los empleados con características similares aparecen agrupados. Esta vista permite identificar los grandes 'arquetipos' de empleados en la organización.")

with col2:
    st.markdown("##### Resumen de los Perfiles")
    if df_sim["Perfil_Empleado"].notna().all():
        for perfil in sorted(df_sim["Perfil_Empleado"].unique()):
            grupo = df_sim[df_sim["Perfil_Empleado"] == perfil]
            with st.expander(f"**Perfil: '{perfil}'** ({len(grupo)} empleados)"):
                st.markdown(f"- **Riesgo de Abandono Medio**: `{grupo['Prob_Abandono'].mean():.1%}`")
                st.markdown(f"- **Clima Laboral Medio**: `{grupo['Clima_Laboral'].mean():.1f}/5`")
                st.markdown(f"- **Antigüedad Media**: `{grupo['Antigüedad'].mean():.1f} años`")
                st.markdown(f"- **Desempeño Medio**: `{grupo['Desempeño'].mean():.1f}/5`")
                if not grupo['Recomendación'].mode().empty:
                    st.markdown(f"- **Recomendación Clave**: _{grupo['Recomendación'].mode()[0]}_")
    else:
        st.warning("No se pudo generar el resumen de perfiles debido a valores nulos.")

st.header("5. Desglose Detallado por Empleado")
st.markdown("Utiliza el selector para ver el análisis individual de cada empleado.")

selected_id = st.selectbox("Selecciona un ID de Empleado para ver su ficha:", df_sim.index)
if selected_id is not None:
    row = df_sim.loc[selected_id]
    st.markdown(f"### Ficha del Empleado: ID {selected_id}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Departamento**: {row.get('Departamento', 'N/A')}")
        st.markdown(f"**Perfil Asignado**: {row.get('Perfil_Empleado', 'N/A')}")
        st.markdown(f"**Edad**: {row.get('Edad', 'N/A')} años")
        st.markdown(f"**Antigüedad**: {row.get('Antigüedad', 'N/A')} años")
        st.markdown(f"**Contrato**: {row.get('Tipo_Contrato', 'N/A')}")

    with col2:
        st.markdown(f"**Desempeño**: `{row.get('Desempeño', 'N/A')}/5`")
        st.markdown(f"**Clima Laboral**: `{row.get('Clima_Laboral', 'N/A')}/5`")
        st.markdown(f"**Horas Extra (media)**: {row.get('Horas_Extra', 'N/A')}h")
        st.markdown(f"**Bajas Último Año**: {row.get('Bajas_Último_Año', 'N/A')}")
        st.markdown(f"**Promociones (2 años)**: {row.get('Promociones_2_Años', 'N/A')}")

    riesgo_color = "red" if row.get('Prob_Abandono', 0) >= 0.6 else ("orange" if row.get('Prob_Abandono', 0) >= 0.3 else "green")
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <h4 style="color:{riesgo_color};">RIESGO DE ABANDONO ESTIMADO: {row.get('Prob_Abandono', 0):.1%}</h4>
            <p><strong>RECOMENDACIÓN ESTRATÉGICA:</strong> {row.get('Recomendación', 'N/A')}</p>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# LÓGICA DE DESCARGA DE INFORMES
# ==============================================================================

# --- Función para generar informe TXT ---
def generate_txt_report(_df_sim):
    report_content = []
    report_content.append("==========================================================")
    report_content.append("    INFORME ESTRATÉGICO DE ANÁLISIS DE PLANTILLA (IA RRHH)")
    report_content.append("==========================================================")
    report_content.append(f"\nFecha de generación: {datetime.today().strftime('%d/%m/%Y')}\n")
    report_content.append("--- 1. ANÁLISIS GENERAL DE RIESGO DE ABANDONO ---")
    report_content.append(f"Resumen: Riesgo Alto (>60%): {high_risk} empleados | Riesgo Medio (30-60%): {medium_risk} | Riesgo Bajo (<30%): {low_risk}\n")
    report_content.append("\n--- 2. ANÁLISIS DE PERFILES DE EMPLEADOS (CLUSTERING) ---")
    for perfil in sorted(_df_sim["Perfil_Empleado"].unique()):
        grupo = _df_sim[_df_sim["Perfil_Empleado"] == perfil]
        report_content.append(f"\n  >> Perfil: '{perfil}' ({len(grupo)} empleados)")
        report_content.append(f"     - Riesgo Abandono: {grupo['Prob_Abandono'].mean():.1%} | Clima Laboral: {grupo['Clima_Laboral'].mean():.1f}/5 | Antigüedad Media: {grupo['Antigüedad'].mean():.1f} años")
    
    report_content.append("\n\n--- 3. DESGLOSE DETALLADO POR EMPLEADO ---")
    for index, row in _df_sim.iterrows():
        report_content.append("\n" + "-"*50)
        report_content.append(f"ID Empleado: {index} | Departamento: {row['Departamento']} | Perfil: {row['Perfil_Empleado']}")
        report_content.append(f"  - RIESGO DE ABANDONO: {row['Prob_Abandono']:.1%}")
        report_content.append(f"  - RECOMENDACIÓN: {row['Recomendación']}")
    
    report_content.append("\n\n==================== FIN DEL INFORME ====================")
    return "\n".join(report_content).encode('utf-8')

# --- Función para generar PDF con solo las gráficas ---
def generate_pdf_of_graphs(_df_sim, _escenarios_sim, _df_pca):
    os.makedirs("temp_img", exist_ok=True)
    
    # 1. Generar y guardar todas las figuras
    fig1, ax1 = plt.subplots(); sns.histplot(_df_sim['Prob_Abandono'], bins=20, kde=True, ax=ax1, color="skyblue"); ax1.set_title("Distribución del Riesgo de Abandono"); plt.tight_layout(); plt.savefig("temp_img/riesgo.png"); plt.close(fig1)
    fig2, ax2 = plt.subplots(); sns.barplot(x=list(_escenarios_sim.keys()), y=list(_escenarios_sim.values()), palette="viridis", ax=ax2); ax2.set_title("Simulación de Políticas Estratégicas"); plt.tight_layout(); plt.savefig("temp_img/simulacion.png"); plt.close(fig2)
    fig3, ax3 = plt.subplots(); _df_sim.groupby('Departamento')['Prob_Abandono'].mean().sort_values().plot(kind='barh', ax=ax3, color='salmon'); ax3.set_title("Riesgo Medio por Departamento"); plt.tight_layout(); plt.savefig("temp_img/riesgo_depto.png"); plt.close(fig3)
    fig4, ax4 = plt.subplots(); sns.scatterplot(data=_df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2", s=80, ax=ax4); ax4.grid(True); ax4.set_title("Perfiles de Empleados (PCA)"); plt.tight_layout(); plt.savefig("temp_img/pca.png"); plt.close(fig4)
    
    # 2. Crear el PDF e insertar las imágenes
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    figures = {
        "Distribución del Riesgo de Abandono": "temp_img/riesgo.png",
        "Simulación de Políticas Estratégicas": "temp_img/simulacion.png",
        "Riesgo Medio por Departamento": "temp_img/riesgo_depto.png",
        "Visualización de Perfiles de Empleados (PCA)": "temp_img/pca.png"
    }

    for title, path in figures.items():
        if os.path.exists(path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, title, ln=True, align='C')
            pdf.ln(5)
            pdf.image(path, w=180)
        
    return pdf.output(dest='S').encode('latin-1')

# --- Botones de descarga en la barra lateral ---
st.sidebar.markdown("---")
st.sidebar.header("📄 Descargar Informes")

# Botón para descargar informe TXT
st.sidebar.download_button(
    label="Descargar Informe General (.txt)",
    data=generate_txt_report(df_sim),
    file_name=f"informe_general_RRHH_{datetime.today().strftime('%Y%m%d')}.txt",
    mime="text/plain"
)

# Lógica para generar y descargar PDF
if st.sidebar.button("Generar Informe de Gráficas (.pdf)"):
    st.session_state.pdf_report_data = generate_pdf_of_graphs(df_sim, escenarios_sim, df_pca)
    st.sidebar.success("¡Informe PDF generado!")

if 'pdf_report_data' in st.session_state and st.session_state.pdf_report_data:
    st.sidebar.download_button(
        label="✅ Descargar Informe de Gráficas",
        data=st.session_state.pdf_report_data,
        file_name=f"informe_grafico_RRHH_{datetime.today().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )
