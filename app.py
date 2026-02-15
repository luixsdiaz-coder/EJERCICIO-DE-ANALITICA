import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
                               title="<b>1. [FUNNEL] Tasa de Conversión</b>",
                               color_discrete_map=colores_dict), use_container_width=True)

    with col2:
        # 2. Distribución de Puntaje (Boxplot)
        st.plotly_chart(px.box(df_solo_contratados, x='gender', y='score', color='gender',
                               title="<b>2. [BOX] Puntaje de Contratación</b>",
                               color_discrete_map=colores_dict, points="all"), use_container_width=True)

    with col3:
        # 3. Porcentaje de Contratados (Pie)
        st.plotly_chart(px.pie(df_solo_contratados, names='gender', hole=0.4, 
                               title="<b>3. [PIE] Distribución Final</b>",
                               color_discrete_map=colores_dict), use_container_width=True)

    # --- II. MATRIZ DE CORRELACIÓN ---
    st.divider()
    st.header("II. Matriz de Correlación")
    cols_corr = [c for c in variables_raiz if c in df.columns] + ['hiring_decision']
    corr_matrix = df[cols_corr].corr()
    st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         title="<b>Heatmap: Relación entre Variables</b>",
                         color_continuous_scale='RdBu_r'), use_container_width=True)

    # --- III. RADAR DE COMPETENCIAS (PERFIL DE RECLUTAMIENTO) ---
    st.divider()
    st.header("III. Radar de Competencias (Perfil de Reclutamiento)")
    st.write("Visualización de méritos técnicos y trayectoria (excluye el puntaje de la entrevista).")
    
    competencias_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
    competencias_presentes = [c for c in competencias_radar if c in df_solo_contratados.columns]
    
    if competencias_presentes:
        fig_radar = go.Figure()
        for g in df_solo_contratados['gender'].unique():
            df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
            if not df_g.empty:
                valores = [df_g[c].mean() for c in competencias_presentes]
                valores += [valores[0]]
                cats = competencias_presentes + [competencias_presentes[0]]
                fig_radar.add_trace(go.Scatterpolar(r=valores, theta=[c.upper() for c in cats],
                                                   fill='toself', name=g.capitalize(),
                                                   line=dict(color=colores_dict.get(g, '#888'))))

        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df[competencias_presentes].max().max()])),
                                title="<b>Radar de Habilidades Promedio</b>")
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- IV. JERARQUÍA DE IMPORTANCIA ---
    st.divider()
    st.header("IV. Jerarquía de Importancia (Machine Learning)")
    
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
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]

    if list_imps:
        # Orden Global (Pareto e Importancia Agrupada)
        imp_glob = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        lista_orden_desc = imp_glob['Factor'].tolist()

        # Gráfico Agrupado Ordenado
        fig_imp_agrup = go.Figure()
        for i, g in enumerate(['female', 'male', 'other']):
            res_g = [imp_m, imp_h, imp_o][i]
            if res_g is not None:
                s = res_g.set_index('Factor').reindex(lista_orden_desc[::-1]).reset_index().fillna(0)
                fig_imp_agrup.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), 
                                              orientation='h', marker_color=list(colores_dict.values())[i]))
        
        fig_imp_agrup.update_layout(title="<b>Importancia por Género (Orden Mayor a Menor)</b>", barmode='group', height=500)
        st.plotly_chart(fig_imp_agrup, use_container_width=True)

    # --- V. ANÁLISIS DE VARIABLES POR GÉNERO (INDEPENDIENTE) ---
    st.divider()
    st.header("V. Análisis de Variables por Género (Independiente)")
    tabs = st.tabs(["🚺 Mujeres", "🚹 Hombres", "⚧ Otros"])
    
    for i, g in enumerate(['female', 'male', 'other']):
        with tabs[i]:
            res_gen = [imp_m, imp_h, imp_o][i]
            if res_gen is not None:
                df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
                # Ordenamos variables por su peso específico para este género
                variables_ordenadas = res_gen.sort_values('Peso', ascending=False)
                
                for _, fila in variables_ordenadas.iterrows():
                    v = fila['Factor']
                    if v in df_g.columns:
                        v_mean, v_min, v_max = df_g[v].mean(), df_g[v].min(), df_g[v].max()
                        titulo = f"<b>{v.upper()}</b> | Media: {v_mean:.2f} | Min: {v_min} | Max: {v_max}"
                        
                        fig = px.histogram(df_g, x=v, title=titulo, 
                                           color_discrete_sequence=[list(colores_dict.values())[i]], 
                                           text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No hay datos suficientes para analizar el género {g}.")

    st.caption("Reporte bajo nomenclatura NIIF para transparencia en Capital Humano.")

else:
    st.info("Por favor, cargue su archivo para iniciar la auditoría.")
