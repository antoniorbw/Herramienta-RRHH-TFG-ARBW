# ==========================================
# Interface Header (added automatically)
# ==========================================
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Herramienta de IA - Planificación Estratégica", layout="centered")

st.title("🧠 Herramienta de IA para la Planificación Estratégica de la Plantilla")
st.markdown(
    """
    **Aplicación desarrollada por *Antonio Wilkinson* como parte de su Trabajo de Fin de Grado (TFG)**  
    Fecha: 24-06-2025

    ---
    **¿Qué hace esta herramienta?**  
    - Predice el riesgo de abandono de cada empleado.  
    - Agrupa perfiles mediante PCA y *clustering*.  
    - Simula políticas estratégicas (formación, salario y combinadas).  
    - Analiza el clima laboral por departamento.  

    Al finalizar, podrás descargar:  
    - Un **informe completo (.txt)** con explicaciones detalladas.  
    - Un **informe gráfico (.pdf)** con las visualizaciones clave.
    """
)

st.markdown("---")


# 📁 Carga de datos
try:
    uploaded = files.upload()
    df = pd.read_csv(next(iter(uploaded)), sep=None, engine='python')
    print("✅ Archivo cargado correctamente.")
except:
    print("⚠️ No se subió archivo. Usando datos de ejemplo.")
    df = pd.read_csv('/mnt/data/datos_empleados_avanzado.csv', sep=';')

# 📦 Librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from IPython.display import HTML
from google.colab import files

# 🔄 Preprocesamiento de datos
df_encoded = pd.get_dummies(df, columns=["Departamento", "Tipo_Contrato"], drop_first=True)
X = df_encoded.drop("Riesgo_Abandono", axis=1)
y = df_encoded["Riesgo_Abandono"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🤖 Entrenamiento del modelo
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Crear carpeta de imágenes si no existe
os.makedirs("temp_img", exist_ok=True)

# --- Gráfica 1: Distribución del Riesgo de Abandono ---
plt.figure(figsize=(7,5))
sns.histplot(df_sim['Prob_Abandono'], bins=20, kde=True)
plt.title("Distribución del riesgo de abandono")
plt.xlabel("Probabilidad de abandono")
plt.ylabel("Número de empleados")
plt.tight_layout()
plt.savefig("temp_img/graf_riesgo.png")
plt.close()

# --- Gráfica 2: Clima Laboral por Departamento ---
if 'Clima_Laboral' in df_sim.columns:
    plt.figure(figsize=(8,5))
    df_sim.groupby("Departamento")["Clima_Laboral"].mean().plot(kind="bar")
    plt.title("Clima laboral medio por departamento")
    plt.ylabel("Puntuación media")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temp_img/graf_clima.png")
    plt.close()

# --- Gráfica 3: Comparativa de Políticas Estratégicas ---
if 'escenarios' in globals() and isinstance(escenarios, dict) and escenarios:
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()), palette="Set2")
    plt.title("Comparativa de Políticas Estratégicas")
    plt.ylabel("Riesgo medio estimado")
    plt.tight_layout()
    plt.savefig("temp_img/graf_politicas.png")
    plt.close()

# --- Gráfica 4: Promedio de Abandono por Departamento ---
if 'Departamento' in df_sim.columns:
    plt.figure(figsize=(8,5))
    df_sim.groupby("Departamento")["Prob_Abandono"].mean().plot(kind="bar", color="salmon")
    plt.title("Promedio de Abandono por Departamento")
    plt.ylabel("Riesgo medio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temp_img/graf_abandono_departamento.png")
    plt.close()

# --- Gráfica 5: Visualización de Perfiles (PCA + Clustering) ---
try:
    clustering_features = ["Edad", "Antigüedad", "Clima_Laboral", "Desempeño", "Prob_Abandono"]
    df_cluster = df_sim[clustering_features].dropna()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_cluster)

    df_cluster["PCA1"] = pca_result[:, 0]
    df_cluster["PCA2"] = pca_result[:, 1]
    df_cluster["Cluster"] = df_sim["Perfil_Empleado"]

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_cluster, x="PCA1", y="PCA2", hue="Cluster", palette="tab10")
    plt.title("Visualización de Perfiles de Empleados (PCA + Clustering)")
    plt.tight_layout()
    plt.savefig("temp_img/clustering_pca.png")
    plt.close()
except Exception as e:
    print("Error al generar la gráfica de PCA:", e)

# 📄 Generar informe PDF para descargar
from fpdf import FPDF
from google.colab import files
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re # Importar la librería re para expresiones regulares

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

        # Eliminar emojis de la recomendación usando una expresión regular
        # Esto asegura que solo se incluyan caracteres que fpdf pueda manejar.
        recommendation_text = row.get('Recomendación', 'No disponible')
        recommendation_text = re.sub(r'[^\x00-\x7F]+', '', recommendation_text) # Elimina caracteres no ASCII

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

# Crear y guardar PDF
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Assuming df_sim and other variables like escenarios are defined from previous cells
# If scenarios is not defined, you might need to define a dummy or calculate it
# for the "Simulación de Políticas" section to avoid errors there.
if 'escenarios' not in globals():
    # Define a placeholder if scenarios wasn't calculated earlier
    escenarios = {
        'Original': df["Riesgo_Abandono"].mean() if 'df' in globals() and 'Riesgo_Abandono' in df.columns else 0,
        'Simulación Combinada': df_sim["Prob_Abandono"].mean() if 'df_sim' in globals() and 'Prob_Abandono' in df_sim.columns else 0
    }


# --- PORTADA ---
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Informe de Análisis Estratégico y de Plantilla", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, f"""
Este informe ha sido generado mediante Inteligencia Artificial aplicada a Recursos Humanos.
Contiene un análisis detallado del riesgo de abandono, clima laboral, simulación de políticas
estratégicas y agrupación de perfiles mediante clustering.

Fecha: {datetime.today().strftime('%d/%m/%Y')}
""")

# 3. Crear carpeta temporal para guardar imágenes
os.makedirs("temp_img", exist_ok=True)

# --- SECCIÓN 1: Riesgo de Abandono ---
pdf.add_page() # Start new section on a new page for clarity
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "1. Riesgo de Abandono", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se ha estimado la probabilidad de abandono utilizando un modelo de aprendizaje automático basado
en características como edad, antigüedad, satisfacción y departamento.

A continuación se muestran los 10 empleados con mayor riesgo estimado y las recomendaciones asociadas.
""")

# Tabla de empleados con mayor riesgo
top_empleados = df_sim.sort_values("Prob_Abandono", ascending=False).head(10)

for i, row in top_empleados.iterrows():
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Empleado {i+1}", ln=True)
    pdf.set_font("Arial", size=12)

    # Clean recommendation text for top employees table as well
    recommendation_text_top = row.get('Recomendación', 'No disponible')
    recommendation_text_top = re.sub(r'[^\x00-\x7F]+', '', recommendation_text_top) # Elimina caracteres no ASCII


    pdf.multi_cell(0, 10, f"""
Edad: {row['Edad']} | Antigüedad: {row['Antigüedad']} años | Departamento: {row['Departamento']}
Clima Laboral: {row['Clima_Laboral']} / 5 | Riesgo estimado: {row['Prob_Abandono']:.2f}
Recomendación: {recommendation_text_top}
""")

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
pdf.add_page() # New page for the graph and its explanation
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Gráfica: Distribución del riesgo", ln=True)
pdf.image(riesgo_path, w=180)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Esta gráfica muestra la distribución de la probabilidad de abandono entre todos los empleados.
El eje X representa la probabilidad (de 0 a 1) y el eje Y indica cuántos empleados tienen esa probabilidad.
Un pico alto cerca de 0.8 indica muchos empleados con alto riesgo, lo que es preocupante.
""")

# --- SECCIÓN 2: Clima Laboral ---
pdf.add_page() # New section on a new page
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "2. Clima Laboral Global", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se analiza la media de satisfacción, motivación y equilibrio vida-trabajo por departamento.
""")

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
    pdf.image(clima_path, w=180)
    pdf.multi_cell(0, 10, """
    Esta gráfica muestra la media de la puntuación de Clima Laboral por departamento.
    Eje X: departamentos. Eje Y: puntuación media de Clima Laboral (de 0 a 5, basado en sus datos).
    Permite identificar áreas con problemas o fortalezas en el ambiente de trabajo.
    """)
else:
    pdf.multi_cell(0, 10, "Datos de Clima Laboral detallado no disponibles en el dataset.")


# --- SECCIÓN 3: Simulación de Medidas Estratégicas ---
pdf.add_page() # New section on a new page
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "3. Simulación de Políticas", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se comparan tres escenarios: mejora de formación, mejora salarial, y combinación de ambas.
El objetivo es ver su impacto en la probabilidad media de abandono.
""")

if 'escenarios' in globals() and isinstance(escenarios, dict) and escenarios:
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()))
    plt.title("Comparativa de políticas estratégicas")
    plt.ylabel("Probabilidad media de abandono")
    plt.tight_layout()
    sim_path = "temp_img/simulacion.png"
    plt.savefig(sim_path)
    plt.close()
    pdf.image(sim_path, w=160)
    pdf.multi_cell(0, 10, """
    Esta gráfica muestra la efectividad de distintas políticas para reducir el riesgo global.
    Eje X: tipo de política aplicada. Eje Y: riesgo medio resultante.
    La política más baja indica la estrategia más efectiva.
    """)
else:
     pdf.multi_cell(0, 10, "Resultados de simulación de políticas no disponibles.")


# --- SECCIÓN 4: Clustering de Perfiles ---
pdf.add_page() # New section on a new page
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "4. Agrupación de Perfiles (Clustering)", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se han agrupado empleados con características similares para facilitar la toma de decisiones adaptadas.
""")

if 'Perfil_Empleado' in df_sim.columns:
    for c in sorted(df_sim["Perfil_Empleado"].unique()):
        grupo = df_sim[df_sim["Perfil_Empleado"] == c]
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Grupo: {c} - {len(grupo)} empleados", ln=True)
        pdf.set_font("Arial", size=12)

        # Include metrics that are available in df_sim
        edad_media = grupo['Edad'].mean() if 'Edad' in grupo.columns else 'N/A'
        antiguedad_media = grupo['Antigüedad'].mean() if 'Antigüedad' in grupo.columns else 'N/A'
        clima_medio = grupo['Clima_Laboral'].mean() if 'Clima_Laboral' in grupo.columns else 'N/A'
        prob_abandono_media = grupo['Prob_Abandono'].mean() if 'Prob_Abandono' in grupo.columns else 'N/A'
        recomendacion_principal = grupo['Recomendación'].mode().values[0] if 'Recomendación' in grupo.columns and not grupo['Recomendación'].mode().empty else 'No disponible'

        # Clean recommendation text for cluster summary as well
        recommendation_principal_cleaned = re.sub(r'[^\x00-\x7F]+', '', str(recomendacion_principal))


        pdf.multi_cell(0, 10, f"""
    Edad media: {edad_media:.1f} | Antigüedad media: {antiguedad_media:.1f}
    Clima Laboral medio: {clima_medio:.1f}
    Probabilidad de Abandono media: {prob_abandono_media:.2f}
    Recomendación principal: {recommendation_principal_cleaned}
    """)
else:
     pdf.multi_cell(0, 10, "Datos de Clustering de perfiles no disponibles.")


# --- SECCIÓN 5: Contenido del Informe Estratégico (Integrado) ---
pdf.add_page() # New section on a new page
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "5. Estrategia General y Conclusiones", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
A partir del análisis anterior, se extraen conclusiones clave:

- Identificar y retener talento joven con alto potencial y riesgo de fuga.
- Fomentar la formación en áreas con baja motivación.
- Usar la segmentación por clúster para implementar medidas personalizadas.
- Promover entrevistas de seguimiento y bienestar en departamentos con bajo clima.
- Repetir este análisis periódicamente y comparar la evolución.

Este informe unifica datos y estrategias para apoyar una planificación de RRHH basada en evidencias.
""")


# Add section for individual employee details with recommendations
pdf.add_page() # New section on a new page
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "6. Detalles por Empleado y Recomendaciones", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Aquí se presenta un listado con el riesgo de abandono estimado y la recomendación
específica generada para cada empleado.
""")

# Iterate through all employees in df_sim to add their individual details
for _, row in df_sim.iterrows():
    # Ensure enough space for the next employee's section, start a new page if needed
    if pdf.get_y() + 40 > pdf.h - pdf.b_margin: # Check if remaining space is less than estimated section height
        pdf.add_page()

    pdf.add_employee_section(row)


# --- EXPORTACIÓN FINAL ---
pdf_path_final = "Informe_IA_RRHH_COMPLETO_FINAL.pdf"
pdf.output(pdf_path_final)
files.download(pdf_path_final)

# Clean up temporary images
if os.path.exists("temp_img"):
    shutil.rmtree("temp_img")

from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import files

# Crear carpeta de imágenes si no existe
os.makedirs("temp_img", exist_ok=True)

# --- Gráfica 1: Distribución del Riesgo de Abandono ---
plt.figure(figsize=(7,5))
sns.histplot(df_sim['Prob_Abandono'], bins=20, kde=True)
plt.title("Distribución del riesgo de abandono")
plt.xlabel("Probabilidad de abandono")
plt.ylabel("Número de empleados")
plt.tight_layout()
plt.savefig("temp_img/graf_riesgo.png")
plt.close()

# --- Gráfica 2: Clima Laboral por Departamento ---
if 'Clima_Laboral' in df_sim.columns:
    plt.figure(figsize=(8,5))
    df_sim.groupby("Departamento")["Clima_Laboral"].mean().plot(kind="bar")
    plt.title("Clima laboral medio por departamento")
    plt.ylabel("Puntuación media")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temp_img/graf_clima.png")
    plt.close()

# --- Gráfica 3: Comparativa de Políticas Estratégicas ---
if 'escenarios' in globals() and isinstance(escenarios, dict) and escenarios:
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()), palette="Set2")
    plt.title("Comparativa de Políticas Estratégicas")
    plt.ylabel("Riesgo medio estimado")
    plt.tight_layout()
    plt.savefig("temp_img/graf_politicas.png")
    plt.close()

# --- Gráfica 4: Promedio de Abandono por Departamento ---
if 'Departamento' in df_sim.columns:
    plt.figure(figsize=(8,5))
    df_sim.groupby("Departamento")["Prob_Abandono"].mean().plot(kind="bar", color="salmon")
    plt.title("Promedio de Abandono por Departamento")
    plt.ylabel("Riesgo medio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temp_img/graf_abandono_departamento.png")
    plt.close()

# --- Gráfica 5: Visualización de Perfiles (PCA + Clustering) ---
if not os.path.exists("temp_img/clustering_pca.png"):
    try:
        from sklearn.decomposition import PCA
        import numpy as np

        clustering_features = ["Edad", "Antigüedad", "Clima_Laboral", "Desempeño", "Prob_Abandono"]
        df_cluster = df_sim[clustering_features].dropna()

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_cluster)

        df_cluster["PCA1"] = pca_result[:, 0]
        df_cluster["PCA2"] = pca_result[:, 1]
        df_cluster["Cluster"] = df_sim["Perfil_Empleado"].values[:len(df_cluster)]

        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df_cluster, x="PCA1", y="PCA2", hue="Cluster", palette="tab10")
        plt.title("Visualización de Perfiles de Empleados (PCA + Clustering)")
        plt.tight_layout()
        plt.savefig("temp_img/clustering_pca.png")
        plt.close()
    except Exception as e:
        print("Error al generar PCA:", e)

# --- Generar PDF solo con las gráficas ---
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Informe Visual - Gráficas Clave del Análisis", ln=True)

def add_graph(title, path):
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, title, ln=True)
    if os.path.exists(path):
        pdf.image(path, w=180)
    else:
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, f"{title} - No disponible")

# Insertar todas las gráficas
add_graph("Distribución del Riesgo de Abandono", "temp_img/graf_riesgo.png")
add_graph("Clima Laboral Medio por Departamento", "temp_img/graf_clima.png")
add_graph("Comparativa de Políticas Estratégicas", "temp_img/graf_politicas.png")
add_graph("Promedio de Abandono por Departamento", "temp_img/graf_abandono_departamento.png")
add_graph("Visualización de Perfiles de Empleados (PCA + Clustering)", "temp_img/clustering_pca.png")

# Guardar y descargar PDF final
pdf.output("Informe_IA_RRHH_SOLO_GRAFICAS.pdf")
files.download("Informe_IA_RRHH_SOLO_GRAFICAS.pdf")

# 📄 Generar informe completo en TXT (explicativo, detallado, con recomendaciones)import osfrom datetime import datetimefecha_actual = datetime.today().strftime('%Y%m%d')nombre_archivo_txt = f"Informe_IA_RRHH_COMPLETO_TXT_{fecha_actual}.txt"lineas_txt = []lineas_txt.append("INFORME COMPLETO DE RECURSOS HUMANOS BASADO EN IA\n")lineas_txt.append("Herramienta creada por Antonio Wilkinson\n")lineas_txt.append(f"Fecha de generación: {datetime.today().strftime('%d/%m/%Y')}\n")lineas_txt.append("="*80 + "\n")# Promedio general de riesgoriesgo_medio = df_sim["Prob_Abandono"].mean()lineas_txt.append(f"Probabilidad media de abandono en la empresa: {riesgo_medio:.2%}\n\n")# Riesgo medio por departamentolineas_txt.append(">> Riesgo medio de abandono por departamento:\n")riesgo_por_dep = df_sim.groupby("Departamento")["Prob_Abandono"].mean()for d, v in riesgo_por_dep.items():    lineas_txt.append(f"  - {d}: {v:.2%}\n")lineas_txt.append("\n")# Segmentación de perfiles si existeif "Perfil_Empleado" in df_sim.columns:    lineas_txt.append(">> Segmentación de perfiles basada en clustering:\n")    for perfil in sorted(df_sim["Perfil_Empleado"].unique()):        grupo = df_sim[df_sim["Perfil_Empleado"] == perfil]        edad_media = grupo["Edad"].mean()        antiguedad_media = grupo["Antigüedad"].mean()        clima_medio = grupo["Clima_Laboral"].mean()        desempeño_medio = grupo["Desempeño"].mean()        riesgo_medio_perfil = grupo["Prob_Abandono"].mean()        lineas_txt.append(f"Perfil {perfil}: Edad media: {edad_media:.1f}, Antigüedad media: {antiguedad_media:.1f}, Clima: {clima_medio:.1f}, Desempeño: {desempeño_medio:.1f}, Riesgo medio: {riesgo_medio_perfil:.2%}\n")    lineas_txt.append("\n")# Detalles por empleadolineas_txt.append(">> Recomendaciones personalizadas por empleado:\n")for idx, row in df_sim.iterrows():    lineas_txt.append(f"Empleado 10:\n")    lineas_txt.append(f"Edad: {row['Edad']} años | Antigüedad: {row['Antigüedad']} años\n")    lineas_txt.append(f"Departamento: {row['Departamento']} | Clima Laboral: {row['Clima_Laboral']} / 5 | Desempeño: {row['Desempeño']} / 5\n")    lineas_txt.append(f"Tipo de Contrato: {row['Tipo_Contrato']} | Horas Extra: {row['Horas_Extra']}h\n")    lineas_txt.append(f"Bajas en el último año: {row['Bajas_Último_Año']} | Promociones en 2 años: {row['Promociones_2_Años']}\n")    lineas_txt.append(f"Probabilidad estimada de abandono: {row['Prob_Abandono']:.2%}\n")    recomendacion = row.get("Recomendación", "No disponible")    lineas_txt.append(f"Recomendación: {recomendacion}\n")    lineas_txt.append("-"*60 + "\n")# Guardar el archivowith open(nombre_archivo_txt, "w", encoding="utf-8") as f:    f.writelines(lineas_txt)print(f"✅ Informe guardado como: {nombre_archivo_txt}")

# 🧠 Aplicar KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)
df_sim["Perfil_Empleado"] = clusters

#  Visualización con reducción de dimensiones
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_cluster_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Perfil"] = df_sim["Perfil_Empleado"]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2")
plt.title("Visualización de Perfiles de Empleados (PCA + Clustering)")
plt.grid(True)
plt.show()

from sklearn.decomposition import PCA

# Verificamos que se pueda hacer PCA
if 'Perfil_Empleado' in df_sim.columns:
    try:
        features = df_sim.select_dtypes(include='number').drop(columns=['Prob_Abandono'], errors='ignore')
        features = features.dropna()
        pca = PCA(n_components=2)
        components = pca.fit_transform(features)

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=components[:,0], y=components[:,1], hue=df_sim.loc[features.index, 'Perfil_Empleado'], palette='Set2')
        plt.title("Visualización de Perfiles de Empleados (PCA + Clustering)")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.legend(title='Perfil')
        plt.tight_layout()
        plt.savefig("temp_img/pca_clustering.png")
        plt.close()
    except Exception as e:
        print("❌ Error generando PCA clustering:", e)

# Assuming previous cells for data loading, preprocessing, and model training have run successfully.
# df and model, scaler, etc., should be available.

# Generar predicciones de probabilidad en el dataset original
# Aplicar el mismo escalador usado en el entrenamiento a los datos originales
# This step is crucial as it creates 'df_sim' and 'Prob_Abandono'
X_original_scaled = scaler.transform(X) # Reutilizar el escalador entrenado

# Predecir las probabilidades de abandono (clase 1)
prob_abandono_original = model.predict_proba(X_original_scaled)[:, 1]

# Crear una copia del DataFrame original y añadir la columna de probabilidad
df_sim = df.copy()
df_sim["Prob_Abandono"] = prob_abandono_original

# --- BEGIN: Add logic to generate 'Recomendación' column ---
# This is a placeholder logic. You should replace this with the actual
# logic used in your notebook to generate the recommendations.
# Example: Simple recommendation based on probability score
def generate_recommendation(prob):
    if prob >= 0.6:
        return "📌 Alto riesgo: Considerar programas de retención y revisión salarial/condiciones."
    elif prob >= 0.3:
        return "⚠️ Medio riesgo: Monitorear, ofrecer formación o mejoras en clima laboral."
    else:
        return "✅ Bajo riesgo: Mantener condiciones, enfoque en desarrollo profesional."

df_sim['Recomendación'] = df_sim['Prob_Abandono'].apply(generate_recommendation)
# --- END: Add logic to generate 'Recomendación' column ---

# --- Move this cell block UPWARDS in the notebook ---
# This cell should be AFTER you have df_sim available, but BEFORE
# any section (like the simulation or strategic report) tries to use 'pdf'.

!pip install fpdf matplotlib # Ensure fpdf is installed

# 📄 Generar informe PDF para descargar
from fpdf import FPDF
from google.colab import files
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re # Importar la librería re para expresiones regulares

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

        # Eliminar emojis de la recomendación usando una expresión regular
        # Esto asegura que solo se incluyan caracteres que fpdf pueda manejar.
        recommendation_text = row.get('Recomendación', 'No disponible')
        recommendation_text = re.sub(r'[^\x00-\x7F]+', '', recommendation_text) # Elimina caracteres no ASCII

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

# Create and save PDF object - THIS LINE MUST BE RUN BEFORE using 'pdf'
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)
# Add the first page or cover page here if desired before other sections
# pdf.add_page() # Optional: Add a cover page if not done in the main report generation block below

# --- END OF CELL BLOCK TO BE MOVED UP ---


# --- SECCIÓN 3: Simulación de Políticas Estratégicas (Corregida y Ampliada) ---
# This cell should now run AFTER the cell above that defines PDFReport and creates 'pdf'
pdf.add_page() # This line should now work
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "3. Simulación de Políticas Estratégicas", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se han simulado tres posibles políticas para reducir el riesgo de abandono:

1. Mejora de la formación: aumenta la satisfacción y motivación.
2. Mejora salarial: mejora el compromiso y percepción de valor.
3. Combinación de ambas: busca un impacto sinérgico.

A continuación se presentan los resultados agregados y la distribución del riesgo por cada política simulada.
""")

# ⚙️ Crear variables de simulación si no están creadas
# This block requires df_sim to be defined, which is done in the cell above
if 'escenarios' not in globals():
    form = df_sim.copy()
    sal = df_sim.copy()
    both = df_sim.copy()

    form["Prob_Abandono"] *= 0.9  # Reducción del 10%
    sal["Prob_Abandono"] *= 0.85 # Reducción del 15%
    both["Prob_Abandono"] *= 0.8 # Reducción del 20%

    escenarios = {
        'Original': df_sim["Prob_Abandono"].mean(),
        'Formación': form["Prob_Abandono"].mean(),
        'Salario': sal["Prob_Abandono"].mean(),
        'Combinada': both["Prob_Abandono"].mean()
    }

# 📊 Gráfica 1: Comparación del riesgo medio
plt.figure(figsize=(6,4))
sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()), palette="Set2")
plt.title("Comparativa de riesgo medio por política")
plt.ylabel("Probabilidad media de abandono")
plt.tight_layout()
graf1_path = "temp_img/simulacion_barra.png"
plt.savefig(graf1_path)
plt.close()
pdf.image(graf1_path, w=180)
pdf.multi_cell(0, 10, """
Esta gráfica muestra la probabilidad media de abandono tras aplicar cada política.
- Eje X: tipo de política.
- Eje Y: riesgo medio estimado de abandono.

Observamos que la política combinada es la más efectiva, seguida por mejoras salariales y de formación.
""")

# 📈 Gráfica 2: Distribución del riesgo por política
try:
    df_escenarios = pd.DataFrame({
        "Formación": form["Prob_Abandono"],
        "Salario": sal["Prob_Abandono"],
        "Combinada": both["Prob_Abandono"]
    })

    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df_escenarios, fill=True)
    plt.title("Distribución del riesgo por política aplicada")
    plt.xlabel("Probabilidad de abandono")
    plt.ylabel("Densidad de empleados")
    plt.tight_layout()
    graf2_path = "temp_img/simulacion_distribucion.png"
    plt.savefig(graf2_path)
    plt.close()
    pdf.image(graf2_path, w=180)
    pdf.multi_cell(0, 10, """
Esta gráfica compara cómo se distribuye el riesgo entre los empleados bajo cada política.
- Eje X: riesgo de abandono individual.
- Eje Y: densidad o frecuencia de empleados con ese riesgo.

La política combinada desplaza más empleados hacia niveles bajos de riesgo, demostrando mayor efectividad.
""")
except Exception as e: # Catch specific exceptions or print the error for debugging
    print(f"Error generating distribution plot: {e}")
    pdf.multi_cell(0, 10, "No se pudo generar la distribución por políticas debido a que no se encontraron los datos simulados o hubo un error de graficación.")


# ✅ Recomendación estratégica
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Recomendación:", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se recomienda priorizar la política combinada, ya que muestra la mayor reducción promedio del riesgo de abandono.

Para optimizar costes:
- Aplicar formación a perfiles con baja antigüedad o desempeño débil.
- Otorgar mejoras salariales a perfiles clave con alto riesgo o buen desempeño.
- Utilizar la segmentación por clúster para personalizar las políticas por grupo de empleados.

Una aplicación estratégica por perfil maximiza el impacto minimizando el coste.
""")

# Assuming previous cells for data loading, preprocessing, and model training have run successfully.
# df and model, scaler, etc., should be available.

# Generar predicciones de probabilidad en el dataset original
# Aplicar el mismo escalador usado en el entrenamiento a los datos originales
# This step is crucial as it creates 'df_sim' and 'Prob_Abandono'
X_original_scaled = scaler.transform(X) # Reutilizar el escalador entrenado

# Predecir las probabilidades de abandono (clase 1)
prob_abandono_original = model.predict_proba(X_original_scaled)[:, 1]

# Crear una copia del DataFrame original y añadir la columna de probabilidad
df_sim = df.copy()
df_sim["Prob_Abandono"] = prob_abandono_original

# --- BEGIN: Add logic to generate 'Recomendación' column ---
# This is a placeholder logic. You should replace this with the actual
# logic used in your notebook to generate the recommendations.
# Example: Simple recommendation based on probability score
def generate_recommendation(prob):
    if prob >= 0.6:
        return "📌 Alto riesgo: Considerar programas de retención y revisión salarial/condiciones."
    elif prob >= 0.3:
        return "⚠️ Medio riesgo: Monitorear, ofrecer formación o mejoras en clima laboral."
    else:
        return "✅ Bajo riesgo: Mantener condiciones, enfoque en desarrollo profesional."

df_sim['Recomendación'] = df_sim['Prob_Abandono'].apply(generate_recommendation)
# --- END: Add logic to generate 'Recomendación' column ---

# --- Move this cell block UPWARDS in the notebook ---
# This cell should be AFTER you have df_sim available, but BEFORE
# any section (like the simulation or strategic report) tries to use 'pdf'.

!pip install fpdf matplotlib # Ensure fpdf is installed

# 📄 Generar informe PDF para descargar
from fpdf import FPDF
from google.colab import files
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re # Importar la librería re para expresiones regulares

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

        # Eliminar emojis de la recomendación usando una expresión regular
        # Esto asegura que solo se incluyan caracteres que fpdf pueda manejar.
        recommendation_text = row.get('Recomendación', 'No disponible')
        recommendation_text = re.sub(r'[^\x00-\x7F]+', '', recommendation_text) # Elimina caracteres no ASCII

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

# Create and save PDF object - THIS LINE MUST BE RUN BEFORE using 'pdf'
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)
# Add the first page or cover page here if desired before other sections
# pdf.add_page() # Optional: Add a cover page if not done in the main report generation block below

# --- END OF CELL BLOCK TO BE MOVED UP ---


# --- SECCIÓN 3: Simulación de Políticas Estratégicas (Corregida y Ampliada) ---
# This cell should now run AFTER the cell above that defines PDFReport and creates 'pdf'

# Add this line to ensure the directory exists before saving plots
os.makedirs("temp_img", exist_ok=True)

pdf.add_page() # This line should now work
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "3. Simulación de Políticas Estratégicas", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se han simulado tres posibles políticas para reducir el riesgo de abandono:

1. Mejora de la formación: aumenta la satisfacción y motivación.
2. Mejora salarial: mejora el compromiso y percepción de valor.
3. Combinación de ambas: busca un impacto sinérgico.

A continuación se presentan los resultados agregados y la distribución del riesgo por cada política simulada.
""")

# ⚙️ Crear variables de simulación si no están creadas
# This block requires df_sim to be defined, which is done in the cell above
if 'escenarios' not in globals():
    form = df_sim.copy()
    sal = df_sim.copy()
    both = df_sim.copy()

    form["Prob_Abandono"] *= 0.9  # Reducción del 10%
    sal["Prob_Abandono"] *= 0.85 # Reducción del 15%
    both["Prob_Abandono"] *= 0.8 # Reducción del 20%

    escenarios = {
        'Original': df_sim["Prob_Abandono"].mean(),
        'Formación': form["Prob_Abandono"].mean(),
        'Salario': sal["Prob_Abandono"].mean(),
        'Combinada': both["Prob_Abandono"].mean()
    }

# 📊 Gráfica 1: Comparación del riesgo medio
plt.figure(figsize=(6,4))
sns.barplot(x=list(escenarios.keys()), y=list(escenarios.values()), palette="Set2")
plt.title("Comparativa de riesgo medio por política")
plt.ylabel("Probabilidad media de abandono")
plt.tight_layout()
graf1_path = "temp_img/simulacion_barra.png"
plt.savefig(graf1_path)
plt.close()
pdf.image(graf1_path, w=180)
pdf.multi_cell(0, 10, """
Esta gráfica muestra la probabilidad media de abandono tras aplicar cada política.
- Eje X: tipo de política.
- Eje Y: riesgo medio estimado de abandono.

Observamos que la política combinada es la más efectiva, seguida por mejoras salariales y de formación.
""")

# 📈 Gráfica 2: Distribución del riesgo por política
try:
    df_escenarios = pd.DataFrame({
        "Formación": form["Prob_Abandono"],
        "Salario": sal["Prob_Abandono"],
        "Combinada": both["Prob_Abandono"]
    })

    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df_escenarios, fill=True)
    plt.title("Distribución del riesgo por política aplicada")
    plt.xlabel("Probabilidad de abandono")
    plt.ylabel("Densidad de empleados")
    plt.tight_layout()
    graf2_path = "temp_img/simulacion_distribucion.png"
    plt.savefig(graf2_path)
    plt.close()
    pdf.image(graf2_path, w=180)
    pdf.multi_cell(0, 10, """
Esta gráfica compara cómo se distribuye el riesgo entre los empleados bajo cada política.
- Eje X: riesgo de abandono individual.
- Eje Y: densidad o frecuencia de empleados con ese riesgo.

La política combinada desplaza más empleados hacia niveles bajos de riesgo, demostrando mayor efectividad.
""")
except Exception as e: # Catch specific exceptions or print the error for debugging
    print(f"Error generating distribution plot: {e}")
    pdf.multi_cell(0, 10, "No se pudo generar la distribución por políticas debido a que no se encontraron los datos simulados o hubo un error de graficación.")


# ✅ Recomendación estratégica
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Recomendación:", ln=True)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
Se recomienda priorizar la política combinada, ya que muestra la mayor reducción promedio del riesgo de abandono.

Para optimizar costes:
- Aplicar formación a perfiles con baja antigüedad o desempeño débil.
- Otorgar mejoras salariales a perfiles clave con alto riesgo o buen desempeño.
- Utilizar la segmentación por clúster para personalizar las políticas por grupo de empleados.

Una aplicación estratégica por perfil maximiza el impacto minimizando el coste.
""")

# 📈 Distribución de probabilidad de abandono
plt.figure(figsize=(8,5))
sns.histplot(df_sim['Prob_Abandono'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de Probabilidad de Abandono')
plt.xlabel('Probabilidad')
plt.ylabel('Número de empleados')
plt.grid(True)
plt.show()

# 🧮 Promedio de abandono por departamento
dept_avg = df_sim.groupby('Departamento')['Prob_Abandono'].mean().sort_values()
plt.figure(figsize=(10,6))
dept_avg.plot(kind='barh', color='salmon')
plt.title('Promedio de riesgo de abandono por departamento')
plt.xlabel('Riesgo promedio')
plt.grid(axis='x')
plt.show()

if 'Departamento' in df_sim.columns and 'Prob_Abandono' in df_sim.columns:
    try:
        abandono_dep = df_sim.groupby("Departamento")["Prob_Abandono"].mean().sort_values()
        plt.figure(figsize=(8,5))
        abandono_dep.plot(kind='barh', color='salmon')
        plt.title("Promedio de Abandono por Departamento")
        plt.xlabel("Probabilidad media de abandono")
        plt.ylabel("Departamento")
        plt.tight_layout()
        plt.savefig("temp_img/abandono_departamento.png")
        plt.close()
    except Exception as e:
        print("❌ Error generando gráfico de abandono por departamento:", e)

# 📄 Exportar informe como archivo de texto plano (.txt)
txt_path = "Informe_IA_RRHH_RESUMEN_MEJORADO.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("📘 INFORME DE INTELIGENCIA ARTIFICIAL EN RRHH\n")
    f.write("Generado automáticamente\n\n")

    f.write("🔍 1. Introducción\n")
    f.write("Este informe presenta un análisis estratégico de los empleados de la empresa utilizando IA.\n")
    f.write("Incluye riesgo de abandono, clima laboral, simulación de políticas y clustering de perfiles.\n\n")

    f.write("📊 2. Clima Laboral y Riesgo Global\n")
    if 'df_sim' in globals():
        clima_medio = df_sim["Clima_Laboral"].mean() if "Clima_Laboral" in df_sim.columns else None
        riesgo_medio = df_sim["Prob_Abandono"].mean() if "Prob_Abandono" in df_sim.columns else None
        if clima_medio is not None and riesgo_medio is not None:
            f.write(f"Clima laboral promedio: {round(clima_medio, 2)} / 5\n")
            f.write(f"Probabilidad media de abandono: {round(riesgo_medio*100, 1)}%\n\n")
        else:
            f.write("Datos no disponibles.\n\n")

    f.write("👥 3. Recomendaciones por Empleado (Top 10)\n")
    top_empleados = df_sim.sort_values("Prob_Abandono", ascending=False).head(10)
    for i, row in top_empleados.iterrows():
        f.write(f"\nEmpleado {i+1}:\n")
        f.write(f"Edad: {row['Edad']} años\n")
        f.write(f"Antigüedad: {row['Antigüedad']} años\n")
        f.write(f"Departamento: {row['Departamento']}\n")
        f.write(f"Clima Laboral: {row['Clima_Laboral']} / 5\n")
        f.write(f"Probabilidad de abandono: {round(row['Prob_Abandono']*100, 1)}%\n")
        f.write(f"Recomendación: {row['Recomendación']}\n")

    f.write("\n📈 4. Simulación de Políticas Estratégicas\n")
    if 'escenarios' in globals() and escenarios:
        for pol, val in escenarios.items():
            f.write(f"- {pol}: {round(val*100, 1)}% de abandono medio\n")
        f.write("La política combinada es la más efectiva en reducir el riesgo.\n\n")
    else:
        f.write("No se pudieron cargar los resultados de la simulación.\n\n")

    f.write("🔬 5. Agrupación de Perfiles (Clustering)\n")
    if "Perfil_Empleado" in df_sim.columns:
        for c in sorted(df_sim["Perfil_Empleado"].unique()):
            grupo = df_sim[df_sim["Perfil_Empleado"] == c]
            edad_media = grupo["Edad"].mean()
            antiguedad = grupo["Antigüedad"].mean()
            clima = grupo["Clima_Laboral"].mean()
            abandono = grupo["Prob_Abandono"].mean()
            f.write(f"\nGrupo {c} ({len(grupo)} empleados):\n")
            f.write(f"Edad media: {round(edad_media,1)} años | Antigüedad: {round(antiguedad,1)} años\n")
            f.write(f"Clima laboral medio: {round(clima,1)} / 5\n")
            f.write(f"Probabilidad media de abandono: {round(abandono*100,1)}%\n")
    else:
        f.write("No se encontró información de clustering.\n")

    f.write("\n🧠 6. Conclusiones y Estrategias Recomendadas\n")
    f.write("- Focalizar formación e incentivos en empleados con riesgo alto.\n")
    f.write("- Evaluar clima laboral por departamento para corregir focos problemáticos.\n")
    f.write("- Aplicar medidas diferenciadas por grupo de empleados.\n")
    f.write("- Repetir análisis periódicamente y usar IA como herramienta predictiva.\n")

# Descargar
files.download(txt_path)