import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría Equidad NIIF - Streamlit", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento (Protocolo NIIF)")
st.markdown("---")

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Cargar Base de Datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)
    
    # Estandarización de columnas
    df.columns = df.columns.str.lower().str.strip()
    
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']
    
    for col in variables_raiz:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = pd.to_numeric(df['hiring_decision'], errors='coerce').fillna(0).astype(int)

    df_solo_contratados = df[df['hiring_decision'] == 1].copy()
    colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

    # --- I. DIAGNÓSTICO DE EMBUDO Y CONVERSIÓN ---
    st.header("I. Diagnóstico de Embudo y Conversión")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 1. Tasa de Conversión (Funnel)
        df_funnel = []
        for g in df['gender'].unique():
            post = len(df[df['gender'] == g])
            cont = len(df[(df['gender'] == g) & (df['hiring_decision'] == 1)])
            df_funnel.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
            df_funnel.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
        st.plotly_chart(px.funnel(df_funnel, x='Cantidad', y='Etapa', color='Género',
                               title="<b>1. Tasa de Conversión</b>",
                               color_discrete_map=colores_dict), use_container_width=True)

    with col2:
        # 2. Distribución de Puntaje (Boxplot) - Mantenemos Score aquí para auditoría visual
        st.plotly_chart(px.box(df_solo_contratados, x='gender', y='score', color='gender',
                               title="<b>2. Exigencia de Puntaje (Score)</b>",
                               color_discrete_map=colores_dict, points="all"), use_container_width=True)

    with col3:
        # 3. Porcentaje de Contratados (Pie)
        st.plotly_chart(px.pie(df_solo_contratados, names='gender', hole=0.4, 
                               title="<b>3. Distribución Final de Contratados</b>",
                               color_discrete_map=colores_dict), use_container_width=True)

    # --- II. MATRIZ DE CORRELACIÓN ---
    st.divider()
    st.header("II. Matriz de Correlación")
    cols_corr = [c for c in variables_raiz if c in df.columns] + ['hiring_decision']
    corr_matrix = df[cols_corr].corr()
    st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         title="<b>Heatmap: Relación entre Variables y Contratación</b>",
                         color_continuous_scale='RdBu_r'), use_container_width=True)

    # --- III. RADAR DE COMPETENCIAS ---
    st.divider()
    st.header("III. Radar de Competencias (Perfil de Reclutamiento)")
    competencias_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
    comp_presentes = [c for c in competencias_radar if c in df_solo_contratados.columns]
    
    if comp_presentes:
        fig_radar = go.Figure()
        for g in df_solo_contratados['gender'].unique():
            df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
            if not df_g.empty:
                valores = [df_g[c].mean() for c in comp_presentes]
                valores += [valores[0]]
                cats = comp_presentes + [comp_presentes[0]]
                fig_radar.add_trace(go.Scatterpolar(r=valores, theta=[c.upper() for c in cats],
                                                   fill='toself', name=g.capitalize(),
                                                   line=dict(color=colores_dict.get(g, '#888'))))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df[comp_presentes].max().max()])),
                                title="<b>Habilidades Promedio (Sin Score)</b>")
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- IV. JERARQUÍA DE IMPORTANCIA (IA - SIN SCORE) ---
    st.divider()
    st.header("IV. Jerarquía de Importancia (Machine Learning)")
    st.info("Nota: Se ha eliminado la variable 'Score' para identificar los méritos de origen que realmente deciden la contratación.")
    
    vars_ia = [v for v in variables_raiz if v != 'score' and v in df.columns]

    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(datos_gen) < 10 or datos_gen['hiring_decision'].nunique() < 2: return None
        X = pd.get_dummies(datos_gen[vars_ia].fillna(0), drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'Factor': X.columns, 'Peso': model.feature_importances_})
        return res.groupby('Factor')['Peso'].sum().reset_index()

    imps = {g: obtener_importancia(g) for g in ['female', 'male', 'other']}
    list_valid_imps = [v for v in imps.values() if v is not None]

    if list_valid_imps:
        # Orden Global Descendente
        imp_glob = pd.concat(list_valid_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        lista_orden = imp_glob['Factor'].tolist()

        fig_imp = go.Figure()
        for g, color in colores_dict.items():
            if imps[g] is not None:
                s = imps[g].set_index('Factor').reindex(lista_orden[::-1]).reset_index().fillna(0)
                fig_imp.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), 
                                              orientation='h', marker_color=color))
        
        fig_imp.update_layout(title="<b>Importancia de Variables (Ordenada Globalmente)</b>", barmode='group', height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- V. ANÁLISIS DE VARIABLES POR GÉNERO (INDEPENDIENTE) ---
    st.divider()
    st.header("V. Análisis de Variables por Género (Independiente)")
    tabs = st.tabs(["🚺 Mujeres", "🚹 Hombres", "⚧ Otros"])
    
    for i, g in enumerate(['female', 'male', 'other']):
        with tabs[i]:
            if imps[g] is not None:
                df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
                # Variables ordenadas por importancia específica de este género
                vars_ordenadas = imps[g].sort_values('Peso', ascending=False)['Factor']
                
                for v in vars_ordenadas:
                    if v in df_g.columns:
                        mu, mn, mx = df_g[v].mean(), df_g[v].min(), df_g[v].max()
                        st.plotly_chart(px.histogram(df_g, x=v, 
                                           title=f"<b>{v.upper()}</b> | Media: {mu:.2f} | Rango: [{mn} - {mx}]", 
                                           color_discrete_sequence=[list(colores_dict.values())[i]], 
                                           text_auto=True), use_container_width=True)
            else:
                st.warning(f"Datos insuficientes para el análisis de {g}.")

    st.caption("Reporte generado bajo nomenclatura NIIF para Capital Humano.")

else:
    st.info("👋 Bienvenida/o. Por favor, suba el archivo de datos para generar el reporte de auditoría.")
