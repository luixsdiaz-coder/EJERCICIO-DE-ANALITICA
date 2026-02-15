import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Estrategia Global - Auditoría Dinámica", layout="wide")

st.title("📊 Auditoría Estratégica de Talento ($200M Scale-up)")
st.markdown("""
**Análisis de ROI y Equidad:** Utilice los filtros laterales para explorar el comportamiento del embudo y los sesgos en la selección.
""")
st.divider()

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Subir base de datos", type=['csv', 'xlsx'])

if archivo:
    df = pd.read_csv(archivo, sep=';') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.str.lower().str.strip()
    
    # --- FILTROS DINÁMICOS EN BARRA LATERAL ---
    st.sidebar.header("🎯 Parámetros del Reporte")
    
    # Filtro de Decisión (con opción de Todos)
    dict_hiring = {"Contratados": 1, "Rechazados": 0}
    opciones_hiring = st.sidebar.multiselect(
        "Filtrar por Decisión de Contratación:",
        options=list(dict_hiring.keys()),
        default=list(dict_hiring.keys())
    )
    # Traducir selección a valores numéricos
    hiring_values = [dict_hiring[x] for x in opciones_hiring]
    
    # Filtro de Género (con opción de Todos)
    generos_disponibles = df['gender'].unique().tolist()
    opciones_genero = st.sidebar.multiselect(
        "Filtrar por Género:",
        options=generos_disponibles,
        default=generos_disponibles
    )
    
    # --- APLICACIÓN DE FILTROS ---
    df_filtrado = df[
        (df['hiring_decision'].isin(hiring_values)) & 
        (df['gender'].isin(opciones_genero))
    ].copy()
    
    # Procesamiento de variables
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']
    
    for col in variables_raiz:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

    # --- SECCIÓN I: TASA DE CONVERSIÓN (FUNNEL) ---
    st.header("I. Tasa de Conversión y Embudo de Selección")
    
    # Calculamos el funnel sobre el dataset total (para que tenga sentido la conversión)
    df_funnel_resumen = []
    # Usamos df (original) pero filtrado solo por los géneros seleccionados para ver su conversión real
    df_base_funnel = df[df['gender'].isin(opciones_genero)]
    
    for g in opciones_genero:
        post = len(df_base_funnel[df_base_funnel['gender'] == g])
        cont = len(df_base_funnel[(df_base_funnel['gender'] == g) & (df_base_funnel['hiring_decision'] == 1)])
        df_funnel_resumen.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
        df_funnel_resumen.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
    
    df_funnel_plot = pd.DataFrame(df_funnel_resumen)
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.plotly_chart(px.funnel(df_funnel_plot, x='Cantidad', y='Etapa', color='Género',
                               title="Eficiencia del Embudo por Género Seleccionado",
                               color_discrete_map=colores_dict), use_container_width=True)
    with col_f2:
        st.metric("Total Candidatos Analizados", len(df_filtrado))
        if 1 in hiring_values:
            tasa_total = (len(df_base_funnel[df_base_funnel['hiring_decision']==1]) / len(df_base_funnel)) * 100 if len(df_base_funnel) > 0 else 0
            st.metric("Tasa de Conversión Global", f"{tasa_total:.2f}%")

    # --- SECCIÓN II: PERFIL DE SCORE Y COMPETENCIAS ---
    st.divider()
    st.header("II. Distribución de Score y Perfil de Competencias")
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(px.box(df_filtrado, x='gender', y='score', color='gender',
                               title="Exigencia de Score en Grupo Seleccionado",
                               color_discrete_map=colores_dict, points="all"), use_container_width=True)
    
    with c2:
        comp_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
        comp_p = [c for c in comp_radar if c in df_filtrado.columns]
        
        if comp_p:
            fig_radar = go.Figure()
            for g in df_filtrado['gender'].unique():
                df_g = df_filtrado[df_filtrado['gender'] == g]
                if not df_g.empty:
                    valores = [df_g[c].mean() for c in comp_p] + [df_g[comp_p[0]].mean()]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=valores, theta=[c.upper() for c in comp_p + [comp_p[0]]],
                        fill='toself', name=g.capitalize(),
                        line=dict(color=colores_dict.get(g, '#888'))
                    ))
            st.plotly_chart(fig_radar.update_layout(title="Radar de Habilidades Promedio"), use_container_width=True)

    # --- SECCIÓN III: IMPORTANCIA DE IA (GLOBAL) ---
    st.divider()
    st.header("III. Drivers de Decisión (Machine Learning)")
    st.info("Este análisis utiliza los datos de TODOS los candidatos para explicar qué causa la contratación.")
    
    vars_ia = [v for v in variables_raiz if v != 'score' and v in df.columns]

    def get_imp(gen):
        d = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(d) < 10 or d['hiring_decision'].nunique() < 2: return None
        X = pd.get_dummies(d[vars_ia].fillna(0), drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, d['hiring_decision'])
        return pd.DataFrame({'Factor': X.columns, 'Peso': model.feature_importances_}).groupby('Factor')['Peso'].sum().reset_index()

    imps = {g: get_imp(g) for g in opciones_genero}
    list_valid = [v for v in imps.values() if v is not None]

    if list_valid:
        g_imp = pd.concat(list_valid).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        fig_imp = go.Figure()
        for g in opciones_genero:
            if g in imps and imps[g] is not None:
                s = imps[g].set_index('Factor').reindex(g_imp['Factor'].tolist()[::-1]).reset_index().fillna(0)
                fig_imp.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), orientation='h', marker_color=colores_dict.get(g)))
        st.plotly_chart(fig_imp.update_layout(title="¿Qué méritos pesan más en la decisión final?", barmode='group'), use_container_width=True)

else:
    st.info("🚀 CSO: Cargue el archivo para activar el análisis estratégico.")
