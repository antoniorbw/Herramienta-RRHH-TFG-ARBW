
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from datetime import datetime

# Cargar CSV y preparar datos
df = pd.read_csv("datos_empleados_avanzado.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Clima_Laboral": "Clima_Laboral",
    "Formación_Reciente": "Formación_Reciente",
    "Riesgo_Abandono": "Riesgo_Abandono"
})

# Convertir tipos
df["Clima_Laboral"] = df["Clima_Laboral"].astype(float)
df["Riesgo_Abandono"] = df["Riesgo_Abandono"].astype(float)

# 1. Gráfica de riesgo de abandono
plt.figure(figsize=(8,5))
sns.histplot(df["Riesgo_Abandono"], bins=10, kde=True, color="tomato")
plt.title("Distribución del Riesgo de Abandono")
plt.xlabel("Probabilidad de Abandono")
plt.ylabel("Número de Empleados")
plt.tight_layout()
plt.savefig("grafica_riesgo_abandono.png")
plt.close()

# 2. Clima laboral por departamento
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="Departamento", y="Clima_Laboral", palette="coolwarm")
plt.title("Clima Laboral por Departamento")
plt.ylabel("Puntuación de Clima (1-5)")
plt.xlabel("Departamento")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafica_clima_departamento.png")
plt.close()

# 3. Simulación de políticas
sim_politicas = pd.DataFrame({
    "Política": ["Sin acción", "Formación", "Salario", "Formación + Salario"],
    "Reducción estimada (%)": [0, 12.5, 18.0, 27.3]
})
plt.figure(figsize=(8,5))
sns.barplot(data=sim_politicas, x="Política", y="Reducción estimada (%)", palette="viridis")
plt.title("Impacto Estimado de Políticas Estratégicas")
plt.ylabel("Reducción del Riesgo de Abandono (%)")
plt.xlabel("Estrategia")
plt.tight_layout()
plt.savefig("grafica_politicas_simuladas.png")
plt.close()

# 4. PCA + Clustering
features = ["Edad", "Antigüedad", "Clima_Laboral", "Salario", "Formación_Reciente", "Riesgo_Abandono"]
X = df[features].dropna()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set1", s=50, alpha=0.7)
plt.title("Perfiles de Empleados (PCA + Clustering)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.tight_layout()
plt.savefig("grafica_pca_clustering.png")
plt.close()

# Generar PDF
pdf_path = "informe_rrhh_graficas_completo.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)
width, height = A4
fecha = datetime.now().strftime("%d/%m/%Y")

# Portada
c.setFont("Helvetica-Bold", 20)
c.drawCentredString(width/2, height - 4*cm, "INFORME GRÁFICO DE RRHH")
c.setFont("Helvetica", 14)
c.drawCentredString(width/2, height - 5*cm, "Desarrollado por Antonio Wilkinson")
c.drawCentredString(width/2, height - 6*cm, "TFG - IA aplicada a la planificación estratégica de la plantilla")
c.setFont("Helvetica", 12)
c.drawCentredString(width/2, height - 7*cm, f"Fecha de generación: {fecha}")
c.showPage()

# Función para secciones gráficas
def add_graph_section(titulo, descripcion, imagen_path):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2*cm, titulo)
    c.setFont("Helvetica", 11)
    text = c.beginText(2*cm, height - 3*cm)
    for line in descripcion.split("\n"):
        text.textLine(line)
    c.drawText(text)
    try:
        c.drawImage(imagen_path, 2*cm, height - 18*cm, width=16*cm, height=12*cm)
    except:
        c.drawString(2*cm, height - 10*cm, "[⚠️ Gráfica no encontrada]")
    c.showPage()

# Añadir secciones al PDF
add_graph_section(
    "1. Distribución del Riesgo de Abandono",
    "Esta gráfica muestra la distribución de la probabilidad de abandono entre los empleados.",
    "grafica_riesgo_abandono.png"
)
add_graph_section(
    "2. Clima Laboral por Departamento",
    "Visualiza el clima interno agrupado por áreas funcionales.",
    "grafica_clima_departamento.png"
)
add_graph_section(
    "3. Simulación de Políticas Estratégicas",
    "Comparativa del impacto estimado de formación, aumentos salariales o ambas.",
    "grafica_politicas_simuladas.png"
)
add_graph_section(
    "4. Perfiles de Empleados (PCA + Clustering)",
    "Visualiza los perfiles detectados mediante análisis de componentes principales y agrupamiento.",
    "grafica_pca_clustering.png"
)

# Página final
c.setFont("Helvetica-Bold", 13)
c.drawString(2*cm, height - 2*cm, "Conclusiones:")
c.setFont("Helvetica", 11)
c.drawString(2*cm, height - 3*cm, "Este informe resume visualmente los resultados clave del análisis de RRHH.")
c.drawString(2*cm, height - 4*cm, "Para más detalles, consulte el informe .txt con recomendaciones personalizadas.")
c.save()
