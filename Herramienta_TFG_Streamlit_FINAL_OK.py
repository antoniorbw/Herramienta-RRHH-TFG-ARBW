
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fpdf import FPDF
import os

st.set_page_config(page_title="IA RRHH", layout="wide")

st.title("📊 Herramienta de IA para Planificación Estratégica de la Plantilla")

st.markdown("Carga tu archivo CSV con datos de empleados para generar análisis y recomendaciones personalizadas.")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.success("✅ Archivo cargado correctamente.")

    # Preprocesamiento
    df_encoded = pd.get_dummies(df, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
    X = df_encoded.drop("Riesgo_Abandono", axis=1)
    y = df_encoded["Riesgo_Abandono"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Simulamos probabilidad de abandono
    df["Prob_Abandono"] = modelo.predict_proba(X_scaled)[:, 1]

    # Clima laboral promedio por departamento
    clima_dep = df.groupby("Departamento")["Clima_Laboral"].mean()

    # PCA y clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Recomendaciones
    def generar_recomendacion(fila):
        if fila["Prob_Abandono"] > 0.8:
            return "Alto riesgo: evaluar condiciones laborales y oportunidades de crecimiento."
        elif fila["Prob_Abandono"] > 0.5:
            return "Riesgo moderado: mantener seguimiento y soporte individualizado."
        else:
            return "Bajo riesgo: continuar con condiciones actuales y fomentar desarrollo."

    df["Recomendación"] = df.apply(generar_recomendacion, axis=1)

    st.header("🔍 Visualización de Análisis")

    figuras = []

    # Gráfica 1: Distribución del riesgo
    fig1, ax1 = plt.subplots()
    ax1.hist(df["Prob_Abandono"], bins=10, color='skyblue', edgecolor='black')
    ax1.set_title("Distribución del riesgo de abandono")
    ax1.set_xlabel("Probabilidad")
    ax1.set_ylabel("Número de empleados")
    st.pyplot(fig1)
    figuras.append(fig1)

    # Gráfica 2: Clima laboral por departamento
    fig2, ax2 = plt.subplots()
    clima_dep.plot(kind="bar", ax=ax2, color="salmon")
    ax2.set_title("Clima laboral medio por departamento")
    ax2.set_ylabel("Puntuación media")
    ax2.set_ylim(0, 5)
    st.pyplot(fig2)
    figuras.append(fig2)

    # Gráfica 3: Clustering PCA
    fig3, ax3 = plt.subplots()
    for cluster in df["Cluster"].unique():
        sub = df[df["Cluster"] == cluster]
        ax3.scatter(sub["PCA1"], sub["PCA2"], label=f"Cluster {cluster}")
    ax3.set_title("Clustering de empleados (PCA)")
    ax3.set_xlabel("PCA1")
    ax3.set_ylabel("PCA2")
    ax3.legend()
    st.pyplot(fig3)
    figuras.append(fig3)

    # Botones para exportar informes
    def generar_txt(df):
        nombre_txt = "informe_final.txt"
        with open(nombre_txt, "w", encoding="utf-8") as f:
            for i, fila in df.iterrows():
                f.write(f"Empleado {i+1}\n")
                f.write(f"Departamento: {fila['Departamento']}\n")
                f.write(f"Edad: {fila['Edad']}\n")
                f.write(f"Clima Laboral: {fila['Clima_Laboral']}\n")
                f.write(f"Probabilidad de Abandono: {int(fila['Prob_Abandono']*100)}%\n")
                f.write(f"Recomendación: {fila['Recomendación']}\n")
                f.write("-"*30 + "\n\n")
        return nombre_txt

    def generar_pdf(figuras):
        nombre_pdf = "informe_graficas.pdf"
        pdf = FPDF()
        for fig in figuras:
            pdf.add_page()
            path = "temp.png"
            fig.savefig(path, bbox_inches='tight')
            pdf.image(path, x=10, y=20, w=180)
            os.remove(path)
        pdf.output(nombre_pdf)
        return nombre_pdf

    st.header("📥 Descargar Informes")

    if st.button("📄 Generar TXT con recomendaciones"):
        archivo_txt = generar_txt(df)
        with open(archivo_txt, "rb") as f:
            st.download_button("⬇️ Descargar informe TXT", f, file_name=archivo_txt)

    if st.button("📊 Generar PDF con gráficas"):
        archivo_pdf = generar_pdf(figuras)
        with open(archivo_pdf, "rb") as f:
            st.download_button("⬇️ Descargar informe gráfico", f, file_name=archivo_pdf)
