import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="CSO Strategic Audit - Scale-up $200M", layout="wide")

st.title("🚀 Brief Ejecutivo: Optimización de Hiring y ROI de Talento")
st.markdown("""
**Estrategia Global:** Auditoría de sesgos y eficiencia técnica para la expansión internacional. 
*Análisis bajo nomenclatura NIIF de Capital Humano.*
""")
st.divider()

# --- 1. CARGA Y PROCESAMIENTO ---
archivo = st.sidebar.file_uploader("Subir Base de Datos (CSV/Excel)", type=['csv', 'xlsx'])

if archivo:
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=None, engine='python')
    else:
        df = pd.read_excel(archivo)
    
    df.columns = df.columns.str.lower().str.strip()
    
    variables_raiz = ['age', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies']
    
    for col in variables_raiz + ['hiring_decision']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 2. FILTROS DINÁMICOS ---
    st.sidebar.header("🎯 Controles de Auditoría")
    dict_hiring = {"Contratados": 1, "Rechazados": 0}
    opciones_hiring = st.sidebar.multiselect("Decisión de Contratación:", options=list(dict_hiring.keys()), default=list(dict_hiring.keys()))
    hiring_values = [dict_hiring[x] for x in opciones_hiring]
    
    generos_disponibles = df['gender'].unique().tolist() if 'gender' in df.columns else []
    opciones_genero = st.sidebar.multiselect("Segmento de Género:", options=generos_disponibles, default=generos_disponibles)
    
    df_filtrado = df[(df['hiring_decision'].isin(hiring_values)) & (df['gender'].isin(opciones_genero))].copy()
    colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

    # --- SECCIÓN I: DIAGNÓSTICO INTEGRAL (CONSOLIDADO) ---
    st.header("I. Diagnóstico de Embudo y Calidad del Talento")
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    
    with c1:
        df_funnel_res = []
        df_base_f = df[df['gender'].isin(opciones_genero)]
        for g in opciones_genero:
            post = len(df_base_f[df_base_f['gender'] == g])
            cont = len(df_base_f[(df_base_f['gender'] == g) & (df_base_f['hiring_decision'] == 1)])
            df_funnel_res.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
            df_funnel_res.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
        st.plotly_chart(px.funnel(pd.DataFrame(df_funnel_res), x='Cantidad', y='Etapa', color='Género', title="<b>1. Embudo de Selección</b>", color_discrete_map=colores_dict), use_container_width=True)

    with c2:
        st.plotly_chart(px.box(df_filtrado, x='gender', y='score', color='gender', title="<b>2. Exigencia de Score</b>", color_discrete_map=colores_dict, points="all"), use_container_width=True)

    with c3:
        comp_radar = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
        comp_p = [c for c in comp_radar if c in df_filtrado.columns]
        if comp_p:
            fig_radar = go.Figure()
            for g in df_filtrado['gender'].unique():
                df_g = df_filtrado[df_filtrado['gender'] == g]
                if not df_g.empty:
                    valores = [df_g[c].mean() for c in comp_p] + [df_g[comp_p[0]].mean()]
                    fig_radar.add_trace(go.Scatterpolar(r=valores, theta=[c.upper() for c in comp_p + [comp_p[0]]], fill='toself', name=g.capitalize(), line=dict(color=colores_dict.get(g, '#888'))))
            st.plotly_chart(fig_radar.update_layout(title="<b>3. Perfil de Habilidades</b>"), use_container_width=True)

    # --- SECCIÓN II: CRITERIOS DE SELECCIÓN (MAPA ORIGINAL + MAPA CUADRADO) ---
    st.divider()
    st.header("II. Criterios de Selección: Análisis de Correlación")
    
    cols_corr = [c for c in variables_raiz if c in df.columns]
    full_cols = cols_corr + ['hiring_decision']
    matriz = df_filtrado[full_cols].corr()
    
    # Escala: Celeste (-1), Amarillo (0), Verde (+1)
    custom_colorscale = [
        [0.0, "rgb(173, 216, 230)"],  # -1
        [0.5, "rgb(255, 255, 0)"],    # 0
        [1.0, "rgb(0, 128, 0)"]       # +1
    ]

    col_mapa_orig, col_mapa_sq = st.columns([1.4, 1])

    with col_mapa_orig:
        # 4. Mapa de Calor Original (con separación)
        fig_orig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.12, shared_xaxes=True)
        
        # Bloque principal
        fig_orig.add_trace(go.Heatmap(z=matriz.loc[cols_corr, cols_corr], x=cols_corr, y=cols_corr, 
                                      colorscale=custom_colorscale, zmin=-1, zmax=1, 
                                      text=matriz.loc[cols_corr, cols_corr].round(2), 
                                      texttemplate="%{text}", showscale=False), row=1, col=1)
        # Fila separada
        fig_orig.add_trace(go.Heatmap(z=[matriz.loc['hiring_decision', cols_corr].values], x=cols_corr, y=['hiring_decision'], 
                                      colorscale=custom_colorscale, zmin=-1, zmax=1, 
                                      text=[matriz.loc['hiring_decision', cols_corr].round(2).values], 
                                      texttemplate="%{text}"), row=2, col=1)

        fig_orig.update_layout(height=600, title_text="<b>4. Mapa de Calor de Auditoría (Original)</b>", template="plotly_white")
        st.plotly_chart(fig_orig, use_container_width=True)

    with col_mapa_sq:
        # 5. Mapa de Calor en Cuadrado (Correlación Directa resumida)
        # Presentamos la relación con Hiring Decision de forma compacta y cuadrada
        cor_hiring = matriz[['hiring_decision']].drop('hiring_decision').sort_values(by='hiring_decision', ascending=False)
        
        fig_sq = go.Figure(data=go.Heatmap(
            z=cor_hiring.values,
            x=['Impacto en Contratación'],
            y=cor_hiring.index,
            colorscale=custom_colorscale, zmin=-1, zmax=1,
            text=cor_hiring.values.round(2),
            texttemplate="%{text}",
            colorbar=dict(title="Nivel")
        ))
        # Ajustamos para que se vea más "cuadrado"
        fig_sq.update_layout(height=600, width=400, title_text="<b>5. Resumen de Éxito (Cuadrado)</b>", template="plotly_white")
        st.plotly_chart(fig_sq, use_container_width=True)

    # --- SECCIÓN III: DRIVERS REALES ---
    st.divider()
    st.header("III. Drivers Reales de Contratación")
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
        st.plotly_chart(fig_imp.update_layout(title="<b>6. Jerarquía de Factores Determinantes</b>", barmode='group', height=500), use_container_width=True)

    # --- SECCIÓN IV: ANÁLISIS DE VARIABLES ---
    st.divider()
    st.header("IV. Análisis de Distribución por Variable")
    if list_valid:
        vars_ordenadas = g_imp['Factor'].tolist()
        for i in range(0, len(vars_ordenadas), 2):
            cols_h = st.columns(2)
            for j in range(2):
                if i + j < len(vars_ordenadas):
                    v = vars_ordenadas[i + j]
                    with cols_h[j]:
                        st.plotly_chart(px.histogram(df_filtrado, x=v, color='gender', barmode='group', title=f"Distribución: {v.upper()}", color_discrete_map=colores_dict, text_auto=True), use_container_width=True)

else:
    st.info("🚀 CSO: Cargue el archivo para activar el análisis dinámico.")
