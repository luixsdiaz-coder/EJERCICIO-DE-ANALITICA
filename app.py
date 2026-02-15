import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Estrategia Global - Auditoría de Talento", layout="wide")

st.title("📊 Brief Ejecutivo: Auditoría Dinámica de Contratación")
st.markdown("""
**Análisis de Sensibilidad:** Compara el perfil de los candidatos aceptados vs. rechazados para identificar sesgos sistemáticos.
""")
st.divider()

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Subir base de datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    df = pd.read_csv(archivo, sep=';') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.str.lower().str.strip()
    
    # --- FILTRO DINÁMICO (LA CLAVE) ---
    st.sidebar.header("🎯 Filtros de Análisis")
    opcion_hiring = st.sidebar.selectbox(
        "Ver candidatos por decisión:",
        options=[1, 0],
        format_func=lambda x: "CONTRATADOS ✅" if x == 1 else "NO CONTRATADOS ❌"
    )
    
    # Creamos el DataFrame filtrado que alimentará a todas las gráficas
    df_filtrado = df[df['hiring_decision'] == opcion_hiring].copy()
    
    # --- PROCESAMIENTO DE VARIABLES ---
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']
    
    for col in variables_raiz:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

    # --- SECCIÓN I: DIAGNÓSTICO VISUAL ---
    st.header(f"I. Perfil de Candidatos ({'Contratados' if opcion_hiring == 1 else 'Rechazados'})")
    c1, c2 = st.columns(2)
    
    with c1:
        # Boxplot Dinámico
        st.plotly_chart(px.box(df_filtrado, x='gender', y='score', color='gender', 
                               title="Distribución de Score por Género", 
                               color_discrete_map=colores_dict, points="all"), use_container_width=True)
    with c2:
        # Composición de Género Dinámica
        st.plotly_chart(px.pie(df_filtrado, names='gender', hole=0.4, 
                               title="Composición de Género en esta categoría", 
                               color_discrete_map=colores_dict), use_container_width=True)

    # --- SECCIÓN II: RADAR COMPARATIVO ---
    st.divider()
    st.header("II. Radar de Competencias del Grupo Seleccionado")
    comp_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
    comp_p = [c for c in comp_radar if c in df_filtrado.columns]
    
    if comp_p:
        fig_radar = go.Figure()
        for g in df_filtrado['gender'].unique():
            df_g = df_filtrado[df_filtrado['gender'] == g]
            if not df_g.empty:
                valores = [df_g[c].mean() for c in comp_p] + [df_g[comp_p[0]].mean()]
                fig_radar.add_trace(go.Scatterpolar(
                    r=valores, 
                    theta=[c.upper() for c in comp_p + [comp_p[0]]], 
                    fill='toself', name=g.capitalize(), 
                    line=dict(color=colores_dict.get(g, '#888'))
                ))
        st.plotly_chart(fig_radar.update_layout(title="Habilidades Promedio"), use_container_width=True)

    # --- SECCIÓN III: JERARQUÍA DE IMPORTANCIA (PARA EL TOTAL DEL DATASET) ---
    st.divider()
    st.header("III. ¿Qué factores determinan la decisión final? (IA)")
    st.info("Este cálculo utiliza el dataset completo para entender por qué unos son aceptados y otros no.")
    
    vars_ia = [v for v in variables_raiz if v != 'score' and v in df.columns]

    def get_imp(gen):
        d = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(d) < 10 or d['hiring_decision'].nunique() < 2: return None
        X = pd.get_dummies(d[vars_ia].fillna(0), drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, d['hiring_decision'])
        return pd.DataFrame({'Factor': X.columns, 'Peso': model.feature_importances_}).groupby('Factor')['Peso'].sum().reset_index()

    imps = {g: get_imp(g) for g in ['female', 'male', 'other']}
    list_valid = [v for v in imps.values() if v is not None]

    if list_valid:
        g_imp = pd.concat(list_valid).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        fig_imp = go.Figure()
        for g, color in colores_dict.items():
            if imps[g] is not None:
                s = imps[g].set_index('Factor').reindex(g_imp['Factor'].tolist()[::-1]).reset_index().fillna(0)
                fig_imp.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), orientation='h', marker_color=color))
        st.plotly_chart(fig_imp.update_layout(title="Importancia de Variables (Modelo Global)", barmode='group'), use_container_width=True)

    # --- SECCIÓN IV: HISTOGRAMAS DINÁMICOS ---
    st.divider()
    st.header("IV. Análisis de Variables Específicas")
    col_hist1, col_hist2 = st.columns(2)
    
    # Elegimos las dos variables más importantes para mostrar dinámicamente
    var1, var2 = g_imp['Factor'].iloc[0], g_imp['Factor'].iloc[1]
    
    with col_hist1:
        st.plotly_chart(px.histogram(df_filtrado, x=var1, color='gender', barmode='group',
                                   title=f"Distribución de {var1.upper()}", 
                                   color_discrete_map=colores_dict, text_auto=True), use_container_width=True)
    with col_hist2:
        st.plotly_chart(px.histogram(df_filtrado, x=var2, color='gender', barmode='group',
                                   title=f"Distribución de {var2.upper()}", 
                                   color_discrete_map=colores_dict, text_auto=True), use_container_width=True)

else:
    st.info("🚀 CSO: Cargue el archivo para activar el análisis dinámico.")
