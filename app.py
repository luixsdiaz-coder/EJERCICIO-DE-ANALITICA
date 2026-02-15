import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría Integral Equidad NIIF", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento")

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Cargar Base de Datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)
    
    df.columns = df.columns.str.lower().str.strip()
    
    # Limpieza y Conversión Numérica
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']
    
    for col in variables_raiz:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = pd.to_numeric(df['hiring_decision'], errors='coerce').fillna(0).astype(int)

    df_solo_contratados = df[df['hiring_decision'] == 1].copy()
    colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

    # --- SECCIÓN I: DIAGNÓSTICO DE EMBUDO Y CONVERSIÓN ---
    st.header("I. Diagnóstico de Embudo y Conversión")
    col_a, col_b = st.columns(2)
    
    with col_a:
        df_funnel = []
        for g in df['gender'].unique():
            post = len(df[df['gender'] == g])
            cont = len(df[(df['gender'] == g) & (df['hiring_decision'] == 1)])
            df_funnel.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
            df_funnel.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
        
        st.plotly_chart(px.funnel(df_funnel, x='Cantidad', y='Etapa', color='Género',
                               title="<b>1. [EMBUDO] Tasa de Conversión</b>",
                               color_discrete_map=colores_dict), use_container_width=True)

    with col_b:
        # Gráfico Box respecto al puntaje de contratación (Score)
        st.plotly_chart(px.box(df_solo_contratados, x='gender', y='score', color='gender',
                               title="<b>2. [BOXPLOT] Distribución de Puntaje en Contratados</b>",
                               color_discrete_map=colores_dict, points="all"), use_container_width=True)

    # --- SECCIÓN II: ESTADÍSTICAS COMPARATIVAS POR GÉNERO (NUEVA) ---
    st.divider()
    st.header("II. Análisis de Variables por Género (Métricas NIIF)")
    st.write("Desglose detallado de la Media, Mínimo y Máximo para cada factor crítico.")
    
    # Seleccionamos variables clave para el análisis de texto
    vars_analisis = ['age', 'score', 'international_exp', 'programming_exp', 'add_languages']
    
    for var in vars_analisis:
        if var in df_solo_contratados.columns:
            # Cálculo de métricas
            stats = df_solo_contratados.groupby('gender')[var].agg(['mean', 'min', 'max']).reset_index()
            
            st.subheader(f"Variable: {var.upper()}")
            cols_stats = st.columns(len(stats))
            for i, row in enumerate(stats.itertuples()):
                with cols_stats[i]:
                    st.metric(label=f"{row.gender.capitalize()}", value=f"{row.mean:.2f} μ", 
                              help=f"Min: {row.min} | Max: {row.max}")
            
            # Gráfico de barras comparativo de promedios
            fig_bar_avg = px.bar(stats, x='gender', y='mean', color='gender', text_auto='.2f',
                                 title=f"Promedio de {var.upper()} entre Contratados",
                                 color_discrete_map=colores_dict)
            st.plotly_chart(fig_bar_avg, use_container_width=True)

    # --- SECCIÓN III: RADAR DE COMPETENCIAS (SIN SCORE) ---
    st.divider()
    st.header("III. Radar de Competencias (Perfil de Reclutamiento)")
    st.write("Este gráfico excluye el puntaje de la entrevista para enfocarse en la trayectoria del candidato.")
    
    # Competencias sin 'score'
    competencias_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
    competencias_presentes = [c for c in competencias_radar if c in df_solo_contratados.columns]
    
    if competencias_presentes:
        fig_radar = go.Figure()
        for g in df_solo_contratados['gender'].unique():
            df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
            valores = [df_g[c].mean() for c in competencias_presentes]
            valores += [valores[0]]  # Cerrar el radar
            cats = competencias_presentes + [competencias_presentes[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=valores, theta=[c.upper() for c in cats],
                fill='toself', name=g.capitalize(),
                line=dict(color=colores_dict.get(g, '#888'))
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, df[competencias_presentes].max().max()])),
            title="<b>Perfil Promedio: Trayectoria y Habilidades</b>", showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- SECCIÓN IV: MATRIZ DE CORRELACIÓN ---
    st.divider()
    st.header("IV. Matriz de Correlación")
    cols_corr = [c for c in variables_raiz if c in df.columns] + ['hiring_decision']
    corr_matrix = df[cols_corr].corr()
    st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         title="<b>Heatmap de Decisiones vs Atributos</b>",
                         color_continuous_scale='RdBu_r'), use_container_width=True)

    # --- SECCIÓN V: IMPORTANCIA IA Y PARETO (ORDENADO) ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols_p = [c for c in variables_raiz if c in datos_gen.columns]
        X = datos_gen[cols_p].fillna(0)
        X = pd.get_dummies(X, drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols_p if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso')

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    st.divider()
    st.header("V. Jerarquía de Importancia (Machine Learning)")
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        imp_glob = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        
        # Pareto
        imp_glob['Peso_Acum'] = (imp_glob['Peso'].cumsum() / imp_glob['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_glob['Factor'], y=imp_glob['Peso'], name="Impacto", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_glob['Factor'], y=imp_glob['Peso_Acum'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>Análisis de Pareto Global</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption("Auditoría técnica generada bajo criterios de transparencia NIIF.")

else:
    st.info("Cargue su archivo para iniciar el análisis.")
