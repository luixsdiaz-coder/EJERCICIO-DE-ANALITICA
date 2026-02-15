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

    # --- SECCIÓN I: DIAGNÓSTICO ESTRATÉGICO Y EMBUDO ---
    st.header("I. Diagnóstico de Embudo y Conversión")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Gráfico de Embudo (Funnel)
        df_funnel = []
        for g in df['gender'].unique():
            post = len(df[df['gender'] == g])
            cont = len(df[(df['gender'] == g) & (df['hiring_decision'] == 1)])
            df_funnel.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
            df_funnel.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
        
        fig_funnel = px.funnel(df_funnel, x='Cantidad', y='Etapa', color='Género',
                               title="<b>1. [EMBUDO] Tasa de Conversión por Género</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'})
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_b:
        # Gráfico de Torta
        df_solo_contratados = df[df['hiring_decision'] == 1].copy()
        st.plotly_chart(px.pie(df_solo_contratados, names='gender', hole=0.4, 
                               title="<b>2. [TORTA] Distribución de Contratados</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}), use_container_width=True)

    # --- SECCIÓN II: MATRIZ DE CORRELACIÓN ---
    st.divider()
    st.header("II. Matriz de Correlación (Análisis de Sesgos Ocultos)")
    
    # Seleccionamos solo columnas numéricas para la correlación
    cols_corr = variables_raiz + ['hiring_decision']
    corr_matrix = df[cols_corr].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         title="<b>3. [HEATMAP] Correlación entre Variables y Decisión</b>",
                         color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- SECCIÓN III: GRÁFICO DE RADAR ---
    st.divider()
    st.header("III. Radar de Competencias (Perfil Promedio de Contratados)")
    
    competencias = ['score', 'international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp']
    fig_radar = go.Figure()

    colores_radar = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}
    
    for g in df_solo_contratados['gender'].unique():
        df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
        # Normalizamos valores para que el radar sea comparable (escala 0-1 o 0-max)
        valores = [df_g[c].mean() for c in competencias]
        # Cerrar el círculo del radar
        valores += [valores[0]]
        cats = competencias + [competencias[0]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=valores, theta=[c.upper() for c in cats],
            fill='toself', name=g.capitalize(),
            line=dict(color=colores_radar.get(g, '#888'))
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(df[competencias].max())])),
        title="<b>4. [RADAR] Comparativa de Perfil por Género</b>", showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- SECCIÓN IV: CAMPANAS DE GAUSS ---
    st.divider()
    st.header("IV. Distribución Normal (Edad y Score)")
    vars_gauss = ['age', 'score']
    for var in vars_gauss:
        if var in df_solo_contratados.columns:
            fig_gauss = go.Figure()
            for g in df_solo_contratados['gender'].unique():
                data_g = df_solo_contratados[df_solo_contratados['gender'] == g][var].dropna()
                if len(data_g) > 1:
                    mu, std = data_g.mean(), data_g.std()
                    x = np.linspace(data_g.min(), data_g.max(), 100)
                    y = norm.pdf(x, mu, std)
                    fig_gauss.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Gauss {g} (μ={mu:.1f})", 
                                                   line=dict(color=colores_radar.get(g, '#888'), width=3)))
            fig_gauss.update_layout(title=f"<b>5. [GAUSS] Distribución de {var.upper()}</b>", barmode='overlay')
            st.plotly_chart(fig_gauss, use_container_width=True)

    # --- SECCIÓN V: IMPORTANCIA IA (ORDENADA MAYOR A MENOR) ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols_presentes = [c for c in variables_raiz if c in datos_gen.columns]
        X = datos_gen[cols_presentes].fillna(0)
        X = pd.get_dummies(X, drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols_presentes if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso')

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    st.divider()
    st.header("V. Análisis de Importancia de Variables (IA)")
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        orden_desc = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=True).reset_index()
        lista_ordenada = orden_desc['Factor'].tolist()
        fig_imp = go.Figure()
        for i, g in enumerate(['female', 'male', 'other']):
            res = [imp_m, imp_h, imp_o][i]
            if res is not None:
                s = res.set_index('Factor').reindex(lista_ordenada).reset_index().fillna(0)
                fig_imp.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), orientation='h', marker_color=list(colores_radar.values())[i]))
        fig_imp.update_layout(title="<b>6. [IA] Jerarquía de Decisión</b>", barmode='group', height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- SECCIÓN VI: PARETO ---
    st.divider()
    st.header("VI. Análisis de Pareto Global")
    if list_imps:
        imp_glob = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        imp_glob['Peso_Acum'] = (imp_glob['Peso'].cumsum() / imp_glob['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_glob['Factor'], y=imp_glob['Peso'], name="Impacto", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_glob['Factor'], y=imp_glob['Peso_Acum'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>7. [PARETO] Criterios Finales</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption("Reporte bajo estándares NIIF para transparencia en Capital Humano.")

else:
    st.info("Cargue su archivo para iniciar la auditoría.")
