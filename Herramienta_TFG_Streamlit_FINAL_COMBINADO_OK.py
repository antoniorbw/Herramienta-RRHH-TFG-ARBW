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
from fpdf import FPDF
import os
import io

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Herramienta IA Estratégica - RRHH", layout="wide")

# ==========================================
# Funciones de Generación de Informes
# ==========================================
def generate_pdf_of_graphs(df_report, model, feature_names):
    if not os.path.exists("temp_img"):
        os.makedirs("temp_img")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Gráfica 1: Distribución del Riesgo
    fig, ax = plt.subplots(); sns.histplot(df_report['Prob_Abandono'], bins=15, kde=True, ax=ax, color="skyblue"); ax.set_title("Distribucion del Riesgo de Abandono"); plt.tight_layout(); plt.savefig("temp_img/riesgo_dist.png"); plt.close(fig)
    
    # Gráfica 2: Impulsores Clave
    importances = pd.DataFrame(data={'Attribute': feature_names, 'Importance': np.abs(model.coef_[0])}).sort_values(by='Importance', ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(importances['Attribute'], importances['Importance'], color='skyblue'); ax.set_title('Top 10 Impulsores Clave del Riesgo'); plt.tight_layout(); plt.savefig("temp_img/impulsores.png"); plt.close(fig)
    
    # Gráfica 3: Riesgo por Departamento
    fig, ax = plt.subplots(); df_report.groupby('Departamento')['Prob_Abandono'].mean().sort_values().plot(kind='barh', ax=ax, color='salmon'); ax.set_title("Riesgo Medio por Departamento"); plt.tight_layout(); plt.savefig("temp_img/riesgo_depto.png"); plt.close(fig)

    pdf.add_page(); pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "Informe Grafico de Analisis de Plantilla", ln=True, align='C')
    
    for title, path in [("Distribucion del Riesgo de Abandono", "temp_img/riesgo_dist.png"), 
                         ("Impulsores Clave del Riesgo", "temp_img/impulsores.png"), 
                         ("Riesgo Medio por Departamento", "temp_img/riesgo_depto.png")]:
        pdf.add_page(); pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, title, ln=True, align='C'); pdf.image(path, w=180)
        
    return pdf.output(dest='S').encode('latin-1')

def generate_extensive_txt_report(df_report, model, feature_names):
    report_lines = [f"INFORME ESTRATEGICO EXTENSO - {datetime.now().strftime('%d/%m/%Y')}\n{'='*60}\n"]
    report_lines.append(f"Analisis para {len(df_report)} empleados.\n")
    report_lines.append(f"1. KPIs Principales:\n   - Riesgo Medio: {df_report['Prob_Abandono'].mean():.1%}\n   - Clima Laboral Medio: {df_report['Clima_Laboral'].mean():.2f}/5\n")
    
    importances = pd.DataFrame(data={'Attribute': feature_names, 'Importance': np.abs(model.coef_[0])}).sort_values(by='Importance', ascending=False)
    top_feature = importances.iloc[0]['Attribute'].replace('_', ' ')
    report_lines.append(f"2. Impulsor Clave Principal: '{top_feature}'\n")
    
    report_lines.append("3. Desglose por Departamento:")
    for name, group in df_report.groupby('Departamento'):
        report_lines.append(f"   - {name}: {len(group)} empleados, Riesgo Medio: {group['Prob_Abandono'].mean():.1%}")

    report_lines.append("\n4. Desglose Individual de Empleados:\n")
    for index, row in df_report.iterrows():
        report_lines.append(f"{'-'*30}\nID Empleado: {index} | Depto: {row['Departamento']} | Riesgo: {row['Prob_Abandono']:.1%}\nRecomendacion: {row['Recomendacion']}\n")
        
    return "\n".join(report_lines).encode('utf-8')

def generate_summary_txt_report(df_report, model, feature_names):
    report_lines = [f"RESUMEN EJECUTIVO - {datetime.now().strftime('%d/%m/%Y')}\n{'='*60}\n"]
    report_lines.append(f"Analisis para {len(df_report)} empleados.\n")
    report_lines.append(f"PUNTOS CLAVE:\n")
    report_lines.append(f" - Riesgo Medio de Abandono: {df_report['Prob_Abandono'].mean():.1%}")
    
    risk_by_dept = df_report.groupby('Departamento')['Prob_Abandono'].mean().sort_values(ascending=False)
    report_lines.append(f" - Departamento con Mayor Riesgo: {risk_by_dept.index[0]} ({risk_by_dept.iloc[0]:.1%})")

    importances = pd.DataFrame(data={'Attribute': feature_names, 'Importance': np.abs(model.coef_[0])}).sort_values(by='Importance', ascending=False)
    top_feature = importances.iloc[0]['Attribute'].replace('_', ' ')
    report_lines.append(f" - Principal Impulsor del Riesgo: {top_feature}\n")

    report_lines.append("EMPLEADOS DE ALTA PRIORIDAD (Top 3):\n")
    for index, row in df_report.nlargest(3, 'Prob_Abandono').iterrows():
        report_lines.append(f" - ID {index} ({row['Departamento']}): Riesgo del {row['Prob_Abandono']:.1%}")

    return "\n".join(report_lines).encode('utf-8')

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

    st.success(f"✅ Archivo **{uploaded_file.name}** procesado. Se han analizado **{len(df_sim)}** empleados.")

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
    
    # --- Módulo de Exportación en Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Opciones de Exportación")
    report_type = st.sidebar.selectbox("Elige el tipo de informe:", ["Informe de Gráficas (PDF)", "Informe Extenso (TXT)", "Resumen Ejecutivo (TXT)"])
    scope_text = dept_selection if dept_selection != 'Todos' else 'Total'

    if st.sidebar.button("Generar Informe para Descargar"):
        if report_type == "Informe de Gráficas (PDF)":
            report_data = generate_pdf_of_graphs(df_filtered, model, X_train_df.columns)
            st.session_state.download_data = {"data": report_data, "name": f"informe_grafico_{scope_text}.pdf", "mime": "application/pdf"}
        elif report_type == "Informe Extenso (TXT)":
            report_data = generate_extensive_txt_report(df_filtered, model, X_train_df.columns)
            st.session_state.download_data = {"data": report_data, "name": f"informe_extenso_{scope_text}.txt", "mime": "text/plain"}
        elif report_type == "Resumen Ejecutivo (TXT)":
            report_data = generate_summary_txt_report(df_filtered, model, X_train_df.columns)
            st.session_state.download_data = {"data": report_data, "name": f"informe_resumen_{scope_text}.txt", "mime": "text/plain"}

    if 'download_data' in st.session_state and st.session_state.download_data:
        st.sidebar.download_button("✅ ¡Listo! Haz clic para descargar", **st.session_state.download_data)


    # ==========================================
    # ESTRUCTURA DE PESTAÑAS (COMPLETA)
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
                    mean_risk = df_filtered['Prob_Abandono'].mean()
                    if mean_risk > 0.6:
                        st.error(f"**Análisis:** El riesgo medio del grupo es de **{mean_risk:.1%}**, lo que indica una situación preocupante.")
                    else:
                        st.success(f"**Análisis:** El riesgo medio del grupo es de **{mean_risk:.1%}**, lo que sugiere una situación mayormente controlada.")
                    st.markdown("**Recomendaciones:** Si hay un pico significativo en la zona de riesgo alto (>70%), es una señal de alerta que requiere una investigación profunda.")

            with col2:
                st.subheader("Top 5 Empleados con Mayor Riesgo")
                top_5_risk = df_filtered.nlargest(5, 'Prob_Abandono')
                for i, (index, row) in enumerate(top_5_risk.iterrows(), 1):
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
                st.info(f"**Análisis:** El factor más determinante para predecir el abandono en tu empresa es **'{top_feature}'**.")
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
                pca = PCA(n_components=2)
                features_cluster = ["Edad", "Antigüedad", "Desempeño", "Salario", "Formación_Reciente", "Clima_Laboral", "Horas_Extra", "Bajas_Último_Año", "Promociones_2_Años"]
                X_cluster_filtered = df_filtered[features_cluster]
                if len(X_cluster_filtered) > 1:
                    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_cluster_filtered))
                    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"], index=X_cluster_filtered.index)
                    df_pca["Perfil"] = df_filtered["Perfil_Empleado"]
                    fig, ax = plt.subplots(); sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Perfil", palette="Set2", s=80, ax=ax); ax.grid(True)
                    st.pyplot(fig)
            with col2:
                st.markdown("##### Resumen de Perfiles Identificados")
                for perfil in sorted(df_filtered['Perfil_Empleado'].unique()):
                    grupo = df_filtered[df_filtered['Perfil_Empleado'] == perfil]
                    with st.expander(f"**Perfil: '{perfil}'** ({len(grupo)} empleados)"):
                        st.markdown(f"Riesgo medio de **{grupo['Prob_Abandono'].mean():.1%}** y clima de **{grupo['Clima_Laboral'].mean():.1f}/5**.")
                        st.info(f"**Recomendación principal:** {grupo['Recomendación'].mode()[0]}")
            
            st.markdown("---")
            st.subheader("Análisis por Departamento")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clima Laboral Medio")
                fig, ax = plt.subplots(); df_filtered.groupby('Departamento')['Clima_Laboral'].mean().sort_values().plot(kind='barh', ax=ax, color='c'); st.pyplot(fig)
            with col2:
                st.markdown("##### Riesgo de Abandono Medio")
                fig, ax = plt.subplots(); df_filtered.groupby('Departamento')['Prob_Abandono'].mean().sort_values().plot(kind='barh', ax=ax, color='salmon'); st.pyplot(fig)

    # --- PESTAÑA 3: CONSULTA Y SIMULACIÓN ---
    with tab3:
        st.header("Consulta Individual y Simulación de Políticas")
        if df_filtered.empty: st.warning("No hay empleados que coincidan con los filtros.")
        else:
            st.subheader("Análisis Individual Detallado (XAI)")
            st.markdown("Selecciona un empleado para ver su ficha y los factores que determinan su riesgo.")
            
            selected_id = st.selectbox("Selecciona un ID de Empleado:", df_filtered.index, key="employee_selector")
            
            if selected_id is not None:
                row = df_filtered.loc[selected_id]
                riesgo_color = "red" if row.get('Prob_Abandono', 0) >= 0.75 else ("orange" if row.get('Prob_Abandono', 0) >= 0.4 else "green")
                st.markdown(f"""<div style="border: 2px solid {riesgo_color}; padding: 15px; border-radius: 10px; margin-top: 15px; background-color: #f8f9fa;">
                    <h5 style="color:{riesgo_color}; margin-bottom: 5px;">RIESGO DE ABANDONO (ID {selected_id}): {row.get('Prob_Abandono', 0):.1%}</h5>
                    <p style="margin-bottom: 0px;"><strong>RECOMENDACIÓN:</strong> {row.get('Recomendación', 'N/A')}</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("##### ¿Por qué tiene este nivel de riesgo?")
                employee_index_in_original_df = df_sim.index.get_loc(selected_id)
                employee_scaled_data = X_scaled_full[employee_index_in_original_df]
                
                contributions = employee_scaled_data * model.coef_[0]
                feature_contribution = pd.DataFrame({'feature': X_train_df.columns, 'contribution': contributions}).sort_values(by='contribution', ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("🔴 **Factores que AUMENTAN el riesgo:**")
                    for i, rec in feature_contribution.head(3).iterrows(): st.markdown(f"- **{rec['feature'].replace('_', ' ')}**")
                with c2:
                    st.markdown("🟢 **Factores que REDUCEN el riesgo:**")
                    for i, rec in feature_contribution.tail(3).sort_values(by='contribution').iterrows(): st.markdown(f"- **{rec['feature'].replace('_', ' ')}**")
            
            st.markdown("---")
            st.subheader("🕹️ Simulador de Políticas 'What-If'")
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                st.markdown("Ajusta el impacto esperado de cada política para ver cómo afectaría al riesgo del grupo filtrado.")
                form_impact = st.slider("Reducción de riesgo por Formación (%)", 0, 50, 10, key="sim_form_tab3")
                sal_impact = st.slider("Reducción de riesgo por Salario (%)", 0, 50, 15, key="sim_sal_tab3")
            with sim_col2:
                base_risk = df_filtered['Prob_Abandono'].mean()
                form_sim = base_risk * (1 - form_impact / 100)
                sal_sim = base_risk * (1 - sal_impact / 100)
                both_sim = base_risk * (1 - (form_impact + sal_impact) / 100)
                escenarios_sim = {'Estado Actual': base_risk, 'Mejora Formación': form_sim, 'Mejora Salarial': sal_sim, 'Política Combinada': both_sim}
                fig_sim, ax_sim = plt.subplots(); bars = sns.barplot(x=list(escenarios_sim.keys()), y=list(escenarios_sim.values()), palette="viridis", ax=ax_sim)
                for bar in bars.patches: ax_sim.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1%}', ha='center', va='bottom', fontweight='bold')
                ax_sim.set_ylabel("Riesgo Medio de Abandono"); st.pyplot(fig_sim)
            
            with st.expander("Ver Análisis Detallado de Escenarios y Recomendaciones"):
                st.markdown(f"""
                #### Análisis del Escenario: Mejora de Formación
                - **Resultado:** Con una reducción de riesgo del **{form_impact}%**, el riesgo medio del grupo bajaría a **{form_sim:.1%}**.
                - **Recomendación:** Aplicar en perfiles "Potencial Crecimiento" o con bajo "Desempeño".
                ---
                #### Análisis del Escenario: Mejora Salarial
                - **Resultado:** Con una reducción de riesgo del **{sal_impact}%**, el riesgo medio bajaría a **{sal_sim:.1%}**.
                - **Recomendación:** Aplicar en perfiles "Alto Desempeño" en riesgo o para corregir inequidades.
                ---
                #### Análisis del Escenario: Política Combinada
                - **Resultado:** Es la más efectiva, reduciendo el riesgo al **{both_sim:.1%}**.
                - **Recomendación:** Usar para grupos críticos y valiosos, como los perfiles "En Riesgo" con buen desempeño.
                """)

    # --- PESTAÑA 4: GLOSARIO Y METODOLOGÍA ---
    with tab4:
        st.header("📚 Glosario y Metodología")
        st.subheader("Glosario de Términos Clave")
        st.markdown("""
        - **Probabilidad de Abandono:** Porcentaje que indica la probabilidad de que un empleado deje la empresa.
        - **Perfil de Empleado (Cluster):** Grupo de empleados con características similares.
        - **Impulsores Clave (Feature Importance):** Los factores o variables que más peso tienen para el modelo.
        - **Explicabilidad (XAI):** Técnicas que permiten entender por qué el modelo ha tomado una decisión específica para un caso concreto.
        """)
        st.subheader("Metodología del Modelo")
        st.markdown("""
        1.  **Modelo Predictivo:** Se utiliza un modelo de **Regresión Logística**.
        2.  **Modelo de Segmentación:** Se usa un algoritmo de **K-Means Clustering**.
        """)

except Exception as e:
    st.error(f"Se ha producido un error inesperado durante la ejecución: {e}")
