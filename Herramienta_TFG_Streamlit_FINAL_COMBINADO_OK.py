# ==========================================
# Cabecera de la Interfaz (añadido automáticamente)
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fpdf import FPDF
import os
import re
import base64

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Herramienta de IA - Planificación Estratégica", layout="centered")

# --- Título y descripción de la aplicación ---
st.title("🧠 Herramienta de IA para la Planificación Estratégica de la Plantilla")
st.markdown(
    """
    **Aplicación desarrollada como parte de un Trabajo de Fin de Grado (TFG)** Fecha: 24-06-2025

    ---
    **¿Qué hace esta herramienta?** - Predice el riesgo de abandono de cada empleado.  
    - Agrupa perfiles mediante PCA y *clustering*.  
    - Simula políticas estratégicas (formación, salario y combinadas).  
    - Analiza el clima laboral por departamento.  

    Al finalizar, podrás descargar:  
    - Un **informe completo (.txt)** con explicaciones detalladas.  
    - Un **informe gráfico (.pdf)** con las visualizaciones clave.
    """
)

st.markdown("---")

# --- Instrucciones y descarga de plantilla ---
st.markdown(""" 
### 📘 Instrucciones de uso de la Herramienta

Esta herramienta ha sido desarrollada para facilitar la **planificación estratégica de los Recursos Humanos** mediante el uso de Inteligencia Artificial.

---

#### 🧾 ¿Qué necesitas?

Sube un archivo `.csv` con los datos de los empleados. El archivo debe tener al menos las siguientes columnas:

- `Edad`, `Antigüedad`, `Desempeño`, `Salario`, `Formación_Reciente`, `Clima_Laboral`, `Departamento`, `Tipo_Contrato`, `Riesgo_Abandono` (opcional si se recalcula), `Horas_Extra`, `Bajas_Último_Año`, `Promociones_2_Años`.

---

#### 📤 Archivos descargables

- **Informe TXT completo:** explicaciones, recomendaciones, glosario, conclusiones.
- **Informe PDF con gráficas:** visualizaciones clave.
""")

# --- Carga de datos en Streamlit ---
uploaded_file = st.file_uploader("📤 Sube tu archivo CSV con datos de empleados", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.success("✅ Archivo cargado correctamente.")
else:
    st.warning("⚠️ Por favor, sube un archivo CSV para continuar.")
    st.stop()

# --- Validación de columnas ---
required_columns = ['Edad', 'Antigüedad', 'Desempeño', 'Salario', 'Formación_Reciente',
                    'Clima_Laboral', 'Departamento', 'Riesgo_Abandono', 'Horas_Extra', 
                    'Bajas_Último_Año', 'Promociones_2_Años', 'Tipo_Contrato']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"Faltan las siguientes columnas en el archivo: {', '.join(missing)}")
    st.stop()

# --- Preprocesamiento de datos y entrenamiento del modelo ---
df_encoded = pd.get_dummies(df, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
X = df_encoded.drop("Riesgo_Abandono", axis=1)
y = df_encoded["Riesgo_Abandono"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# --- Generación de predicciones y recomendaciones ---
X_original_scaled = scaler.transform(X)
prob_abandono_original = model.predict_proba(X_original_scaled)[:, 1]
df_sim = df.copy()
df_sim["Prob_Abandono"] = prob_abandono_original

def generate_recommendation(prob):
    if prob >= 0.6:
        return "📌 Alto riesgo: Considerar programas de retención y revisión salarial/condiciones."
    elif prob >= 0.3:
        return "⚠️ Riesgo medio: Monitorear, ofrecer formación o mejoras en clima laboral."
    else:
        return "✅ Bajo riesgo: Mantener condiciones, enfoque en desarrollo profesional."
df_sim['Recomendación'] = df_sim['Prob_Abandono'].apply(generate_recommendation)

# --- Clustering de perfiles ---
features = ["Edad", "Antigüedad", "Desempeño", "Salario", "Formación_Reciente", "Clima_Laboral", 
            "Horas_Extra", "Bajas_Último_Año", "Promociones_2_Años"]
X_cluster = df_sim[features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)
df_sim["Perfil_Empleado"] = clusters

perfil_dict = {
    0: "Potencial crecimiento",
    1: "Bajo compromiso",
    2: "Alto desempeño",
    3: "En riesgo"
}
df_sim["Perfil_Empleado"] = df_sim["Perfil_Empleado"].map(perfil_dict)


# --- Creación de carpeta temporal para imágenes ---
os.makedirs("temp_img", exist_ok=True)

# ==========================================
# Generación de informe en PDF
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Informe Estratégico de Plantilla - IA RRHH", ln=True, align="C")
        self.set_font("Arial", "", 10)
        self.cell(0, 10, "Generado automáticamente a partir del análisis de empleados", ln=True, align="C")
        self.ln(5)

    def add_employee_section(self, row):
        self.set_font("Arial", "B", 11)
        self.cell(0, 10, f"Departamento: {row['Departamento']}", ln=True)
        self.set_font("Arial", "", 10)
        recommendation_text = row.get('Recomendación', 'No disponible')
        recommendation_text = re.sub(r'[^\x00-\x7F]+', '', recommendation_text)
        self.multi_cell(0, 8, f"""Edad: {row['Edad']} años
Antigüedad: {row['Antigüedad']} años
Desempeño: {row['Desempeño']} / 5
Clima Laboral: {row['Clima_Laboral']} / 5
Horas Extra: {row['Horas_Extra']}h
Tipo de Contrato: {row['Tipo_Contrato']}
Bajas Último Año: {row['Bajas_Último_Año']}
Promociones en 2 Años: {row['Promociones_2_Años']}
Probabilidad de Abandono: {round(row['Prob_Abandono']*100)}%
Recomendación: {recommendation_text}
""")
        self.ln(3)

def generate_pdf_report():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- PORTADA ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Informe de Análisis Estratégico y de Plantilla", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"""
Este informe ha sido generado mediante Inteligencia Artificial aplicada a Recursos Humanos.
Contiene un análisis detallado del riesgo de abandono, clima laboral, simulación de políticas
estratégicas y agrupación de perfiles mediante clustering.

Fecha: {datetime.today().strftime('%d/%m/%Y')}
""")

    # --- GRÁFICAS ---
    # Gráfica de distribución de riesgo
    plt.figure(figsize=(7,5))
    sns.histplot(df_sim['Prob_Abandono'], bins=20, kde=True)
    plt.title("Distribución del riesgo de abandono")
    plt.xlabel("Probabilidad de abandono")
    plt.ylabel("Número de empleados")
    plt.tight_layout()
    riesgo_path = "temp_img/riesgo.png"
    plt.savefig(riesgo_path)
    plt.close()

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gráfica: Distribución del riesgo", ln=True)
    pdf.image(riesgo_path, w=180)

    # Gráfica de clima laboral
    if 'Clima_Laboral' in df_sim.columns:
        plt.figure(figsize=(8,5))
        df_sim.groupby("Departamento")["Clima_Laboral"].mean().plot(kind="bar")
        plt.title("Clima laboral medio por departamento")
        plt.ylabel("Puntuación media")
        plt.xticks(rotation=45)
        plt.tight_layout()
        clima_path = "temp_img/clima.png"
        plt.savefig(clima_path)
        plt.close()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Gráfica: Clima Laboral", ln=True)
        pdf.image(clima_path, w=180)
        
    # Gráfica de simulación de políticas
    form = df_sim.copy()
    sal = df_sim.copy()
    both = df_sim.copy()
    form["Prob_Abandono"] *= 0.9
    sal["Prob_Abandono"] *= 0.85
    both["Prob_Abandono"] *= 0.8
    escenarios = {
        'Original': df_sim["Prob_Abandono"].mean(),
        'Formación': form["Prob_Abandono"].mean(),
        'Salario': sal["Prob_Abandono"].mean(),
        'Combinada': both["Prob_Abandono"].mean()
    }
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()), palette="Set2")
    plt.title("Comparativa de riesgo medio por política")
    plt.ylabel("Probabilidad media de abandono")
    plt.tight_layout()
    sim_path = "temp_img/simulacion.png"
    plt.savefig(sim_path)
    plt.close()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gráfica: Simulación de Políticas", ln=True)
    pdf.image(sim_path, w=180)

    # Gráfica de PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["Perfil"] = df_sim["Perfil_Empleado"]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2")
    plt.title("Visualización de Perfiles de Empleados (PCA + Clustering)")
    plt.grid(True)
    pca_path = "temp_img/pca_clustering.png"
    plt.savefig(pca_path)
    plt.close()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gráfica: Clustering de Perfiles (PCA)", ln=True)
    pdf.image(pca_path, w=180)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# Generación de informe en TXT
# ==========================================
def generate_txt_report():
    report_content = []
    report_content.append("==========================================================")
    report_content.append("INFORME ESTRATÉGICO DE RECURSOS HUMANOS")
    report_content.append("==========================================================")
    report_content.append(f"Fecha de generación: {datetime.today().strftime('%d/%m/%Y')}\n")
    
    # --- 1. Riesgo de Abandono ---
    report_content.append("--- 1. ANÁLISIS DE RIESGO DE ABANDONO ---")
    high_risk = (df_sim["Prob_Abandono"] >= 0.6).sum()
    medium_risk = ((df_sim["Prob_Abandono"] >= 0.3) & (df_sim["Prob_Abandono"] < 0.6)).sum()
    low_risk = (df_sim["Prob_Abandono"] < 0.3).sum()
    report_content.append(f"\nResumen de niveles de riesgo:")
    report_content.append(f"  - ✅ Riesgo bajo (<30%): {low_risk} empleados")
    report_content.append(f"  - ⚠️ Riesgo medio (30-60%): {medium_risk} empleados")
    report_content.append(f"  - 📌 Riesgo alto (>60%): {high_risk} empleados\n")
    report_content.append("Recomendaciones para empleados con alto riesgo:")
    top_empleados = df_sim.sort_values("Prob_Abandono", ascending=False).head(5)
    for i, row in top_empleados.iterrows():
        report_content.append(f"  - Empleado (ID {i}): Probabilidad del {row['Prob_Abandono']:.1%}. Recomendación: {row['Recomendación']}")
    
    # --- 2. Simulación de Políticas ---
    report_content.append("\n--- 2. SIMULACIÓN DE POLÍTICAS ESTRATÉGICAS ---")
    form = df_sim.copy()
    sal = df_sim.copy()
    both = df_sim.copy()
    form["Prob_Abandono"] *= 0.9
    sal["Prob_Abandono"] *= 0.85
    both["Prob_Abandono"] *= 0.8
    escenarios = {
        'Original': df_sim["Prob_Abandono"].mean(),
        'Formación': form["Prob_Abandono"].mean(),
        'Salario': sal["Prob_Abandono"].mean(),
        'Combinada': both["Prob_Abandono"].mean()
    }
    report_content.append("\nImpacto de las políticas en el riesgo medio de abandono:")
    for k, v in escenarios.items():
        report_content.append(f"  - {k}: {v:.2%}")
    report_content.append("\nConclusión: La política combinada es la más efectiva para reducir el riesgo general.")

    # --- 3. Clustering de Perfiles ---
    report_content.append("\n--- 3. PERFILES DE EMPLEADOS (CLUSTERING) ---")
    report_content.append("\nSe han identificado 4 perfiles de empleados basados en sus características:")
    for perfil in sorted(df_sim["Perfil_Empleado"].unique()):
        grupo = df_sim[df_sim["Perfil_Empleado"] == perfil]
        report_content.append(f"\n  - Perfil '{perfil}': ({len(grupo)} empleados)")
        report_content.append(f"    - Riesgo de abandono medio: {grupo['Prob_Abandono'].mean():.1%}")
        report_content.append(f"    - Clima laboral medio: {grupo['Clima_Laboral'].mean():.1f}/5")
        report_content.append(f"    - Antigüedad media: {grupo['Antigüedad'].mean():.1f} años")

    report_content.append("\n\n--- FIN DEL INFORME ---")
    return "\n".join(report_content)


# --- Visualización en Streamlit y botones de descarga ---
st.markdown("---")
st.header("📊 Resultados del Análisis")

st.subheader("Distribución del Riesgo de Abandono")
fig, ax = plt.subplots()
sns.histplot(df_sim['Prob_Abandono'], bins=10, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Riesgo de Abandono por Departamento")
fig2, ax2 = plt.subplots()
df_sim.groupby('Departamento')['Prob_Abandono'].mean().sort_values().plot(kind='barh', ax=ax2)
st.pyplot(fig2)

st.subheader("Perfiles de Empleados (Clustering con PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Perfil"] = df_sim["Perfil_Empleado"]
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2", ax=ax3)
ax3.grid(True)
st.pyplot(fig3)

st.markdown("---")
st.header("📥 Descargar Informes")

# Botón de descarga para el informe TXT
txt_report_data = generate_txt_report()
st.download_button(
    label="📄 Descargar Informe Completo (.txt)",
    data=txt_report_data,
    file_name=f"informe_estrategico_{datetime.today().strftime('%Y%m%d')}.txt",
    mime="text/plain"
)

# Botón de descarga para el informe PDF
pdf_report_data = generate_pdf_report()
st.download_button(
    label="📈 Descargar Informe con Gráficas (.pdf)",
    data=pdf_report_data,
    file_name=f"informe_grafico_{datetime.today().strftime('%Y%m%d')}.pdf",
    mime="application/pdf"
)

# Limpiar carpeta temporal de imágenes
# shutil.rmtree("temp_img")
