import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría Integral de Equidad NIIF", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento (Análisis Completo)")

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Cargar Base de Datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)
    
    df.columns = df.columns.str.lower().str.strip()
    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = df['hiring_decision'].astype(int)

    # Listado total de variables independientes
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 
                      'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                      'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO DE EMBUDO ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    df_post = df['gender'].value_counts().reset_index(name='cantidad').assign(estado='Postulantes')
    df_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index(name='cantidad').assign(estado='Contratados')
    df_embudo = pd.concat([df_post, df_cont])

    st.plotly_chart(px.bar(df_embudo, x='gender', y='cantidad', color='estado', barmode='group',
                           title="<b>1. Embudo de Selección por Género</b>",
                           color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'}, text_auto=True), use_container_width=True)

    # --- SECCIÓN II: IMPORTANCIA IA (ORDENADA) ---
    st.divider()
    st.header("II. Importancia de Variables (Orden Mayor a Menor)")

    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols = [c for c in variables_raiz if c in datos_gen.columns]
        X = pd.get_dummies(datos_gen[cols], drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso').sort_values('Peso', ascending=True)

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    if imp_h is not None:
        orden = imp_h['Factor'].tolist()
        fig_imp = go.Figure()
        if imp_m is not None: fig_imp.add_trace(go.Bar(y=imp_m.set_index('Factor').reindex(orden).index, x=imp_m.set_index('Factor').reindex(orden)['Peso'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
        fig_imp.add_trace(go.Bar(y=orden, x=imp_h['Peso'], name='Hombres', orientation='h', marker_color='#3d5a80'))
        if imp_o is not None: fig_imp.add_trace(go.Bar(y=imp_o.set_index('Factor').reindex(orden).index, x=imp_o.set_index('Factor').reindex(orden)['Peso'], name='Otros', orientation='h', marker_color='#98c1d9'))
        fig_imp.update_layout(title="<b>2. Criterios de Selección: Comparativa IA</b>", barmode='group', height=600)
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- SECCIÓN III: PARETO GLOBAL ---
    st.divider()
    st.header("III. Análisis de Pareto Global")
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        imp_glob = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        imp_glob['Peso_Acum'] = (imp_glob['Peso'].cumsum() / imp_glob['Glob'].sum() if 'Glob' in imp_glob else imp_glob['Peso'].cumsum() / imp_glob['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_glob['Factor'], y=imp_glob['Peso'], name="Impacto", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_glob['Factor'], y=imp_glob['Peso_Acum'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>3. Pareto: Jerarquía de Decisiones</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    # --- SECCIÓN IV: TODAS LAS VARIABLES INDEPENDIENTES (COMPARATIVAS) ---
    st.divider()
    st.header("IV. Análisis Detallado de TODAS las Variables Independientes")
    st.write("Comparativa del perfil promedio de los **Contratados** por cada variable.")
    
    df_contratados = df[df['hiring_decision'] == 1]
    
    # Generamos una cuadrícula para mostrar todas las variables
    cols_visualizar = [v for v in variables_raiz if v in df.columns]
    
    # Dividimos en filas de 2 gráficos para mejor legibilidad
    for i in range(0, len(cols_visualizar), 2):
        c1, c2 = st.columns(2)
        with c1:
            v = cols_visualizar[i]
            fig = px.histogram(df_contratados, x=v, color='gender', barmode='group', histfunc='avg',
                               title=f"Promedio/Distribución: {v.upper()}",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}, text_auto='.2f')
            st.plotly_chart(fig, use_container_width=True)
        if i + 1 < len(cols_visualizar):
            with c2:
                v2 = cols_visualizar[i+1]
                fig2 = px.histogram(df_contratados, x=v2, color='gender', barmode='group', histfunc='avg',
                                   title=f"Promedio/Distribución: {v2.upper()}",
                                   color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}, text_auto='.2f')
                st.plotly_chart(fig2, use_container_width=True)

    st.caption("Nota: Reporte de Auditoría bajo nomenclatura NIIF para la presentación de resultados de capital humano.")

else:
    st.info("Cargue el archivo para generar la auditoría completa de variables.")
