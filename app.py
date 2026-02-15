import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría de Equidad", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento")

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Cargar Base de Datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    # Lectura del archivo
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)
    
    # Normalización de nombres de columnas
    df.columns = df.columns.str.lower().str.strip()
    
    # Convertir Hiring_decision a entero (1 y 0)
    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = df['hiring_decision'].astype(int)

    # Definición de variables raíz (Aseguramos que coincidan con tu dataset)
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 
                      'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                      'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO DE EMBUDO Y SESGOS ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
    # Preparación de datos para el Gráfico de Embudo
    conteo_post = df['gender'].value_counts().reset_index()
    conteo_post.columns = ['genero', 'cantidad']
    conteo_post['estado'] = 'Postulantes'
    
    conteo_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index()
    conteo_cont.columns = ['genero', 'cantidad']
    conteo_cont['estado'] = 'Contratados'
    
    df_embudo = pd.concat([conteo_post, conteo_cont])

    fig_embudo = px.bar(df_embudo, x='genero', y='cantidad', color='estado', barmode='group',
                        title="<b>1. EMBUDO DE SELECCIÓN: Postulantes vs Contratados</b>",
                        color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'},
                        text_auto=True)
    st.plotly_chart(fig_embudo, use_container_width=True)

    col1, col2 = st.columns(2)
    df_solo_contratados = df[df['hiring_decision'] == 1]

    with col1:
        fig_torta = px.pie(df_solo_contratados, names='gender', hole=0.4, 
                           title="<b>2. Distribución Final de Contratados</b>",
                           color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'})
        st.plotly_chart(fig_torta, use_container_width=True)

    with col2:
        fig_caja = px.box(df_solo_contratados, x='gender', y='score', color='gender', 
                          title="<b>3. Nivel de Exigencia (Solo Contratados)</b>",
                          color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                          points="all")
        st.plotly_chart(fig_caja, use_container_width=True)

    # --- SECCIÓN II: IMPORTANCIA IA ---
    st.divider()
    st.header("II. Análisis de Importancia de Variables")

    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if datos_gen.empty: return None
        # Seleccionamos solo variables que existan en el DF
        cols_finales = [c for c in variables_raiz if c in datos_gen.columns]
        X = pd.get_dummies(datos_gen[cols_finales], drop_first=True)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': modelo.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols_finales if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'})

    imp_m = obtener_importancia('female')
    imp_h = obtener_importancia('male')

    if imp_m is not None and imp_h is not None:
        union = pd.merge(imp_m, imp_h, on='Factor', suffixes=('_Mujeres', '_Hombres')).sort_values('Peso_Mujeres')
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Mujeres'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
        fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Hombres'], name='Hombres', orientation='h', marker_color='#3d5a80'))
        fig_comp.update_layout(title="<b>4. Comparativa de Criterios de Decisión (IA)</b>", barmode='group', height=500)
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    tab_m, tab_h = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino"])
    
    def dibujar_radio(datos_gen, resumen, color, pestana):
        with pestana:
            if resumen is not None:
                df_exito = datos_gen[datos_gen['hiring_decision']==1]
                for i, fila in resumen.sort_values('Peso', ascending=False).head(5).iterrows():
                    var = fila['Factor']
                    fig = px.histogram(df_exito, x=var, 
                                       title=f"Top Factor: {var} (Impacto: {fila['Peso']:.1%})", 
                                       color_discrete_sequence=[color], text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

    dibujar_radio(df[df['gender']=='female'], imp_m, '#e07a5f', tab_m)
    dibujar_radio(df[df['gender']=='male'], imp_h, '#3d5a80', tab_h)

    # --- SECCIÓN IV: VARIABLES INDEPENDIENTES & PARETO ---
    st.divider()
    st.header("IV. Análisis de Variables Independientes (Pareto)")

    if imp_m is not None and imp_h is not None:
        # 1. Pareto Global
        imp_global = pd.concat([imp_m, imp_h]).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        imp_global['Peso_Acumulado'] = (imp_global['Peso'].cumsum() / imp_global['Peso'].sum()) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=imp_global['Factor'], y=imp_global['Peso'], name="Impacto", marker_color='#3d5a80'))
        fig_pareto.add_trace(go.Scatter(x=imp_global['Factor'], y=imp_global['Peso_Acumulado'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f", width=3)))

        fig_pareto.update_layout(
            title="<b>5. PARETO: Peso de Variables Independientes en la Contratación</b>",
            yaxis=dict(title="Importancia IA"),
            yaxis2=dict(title="Acumulado %", overlaying="y", side="right", range=[0, 110]),
            height=500
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # 2. Comparativa Doble Estándar
        st.subheader("Análisis de Competencias Independientes")
        vars_estudio = [v for v in ['international_exp', 'entrepeneur_exp', 'programming_exp', 'add_languages'] if v in df.columns]
        
        if vars_estudio:
            df_comp_indep = df_solo_contratados.groupby('gender')[vars_estudio].mean().reset_index()
            df_comp_melted = df_comp_indep.melt(id_vars='gender', var_name='Variable', value_name='Promedio')

            fig_indep = px.bar(df_comp_melted, x='Variable', y='Promedio', color='gender', barmode='group',
                               title="<b>6. Comparativa de Perfil Técnico (Contratados)</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}, text_auto='.2f')
            st.plotly_chart(fig_indep, use_container_width=True)
    
    st.caption("Nota: Análisis basado en nomenclatura NIIF para la presentación de Estados no Financieros.")

else:
    st.info("Por favor, suba un archivo CSV o Excel para iniciar la auditoría.")
