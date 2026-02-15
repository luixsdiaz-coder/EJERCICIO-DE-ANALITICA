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

    # Definición de variables raíz
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
                           title="<b>2. Composición Final de Contratados</b>",
                           color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'})
        st.plotly_chart(fig_torta, use_container_width=True)

    with col2:
        fig_caja = px.box(df_solo_contratados, x='gender', y='score', color='gender', 
                          title="<b>3. Nivel de Exigencia (Score de Contratados)</b>",
                          color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                          points="all")
        st.plotly_chart(fig_caja, use_container_width=True)

    # --- SECCIÓN II: IMPORTANCIA IA ---
    st.divider()
    st.header("II. Análisis de Importancia de Variables por Género")

    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: 
            return None
        
        cols_finales = [c for c in variables_raiz if c in datos_gen.columns]
        X = pd.get_dummies(datos_gen[cols_finales], drop_first=True)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        
        res = pd.DataFrame({'v': X.columns, 'p': modelo.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols_finales if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'})

    imp_m = obtener_importancia('female')
    imp_h = obtener_importancia('male')
    imp_o = obtener_importancia('other')

    # Gráfico Comparativo de Barras Agrupadas
    fig_comp = go.Figure()
    if imp_m is not None:
        fig_comp.add_trace(go.Bar(y=imp_m['Factor'], x=imp_m['Peso'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
    if imp_h is not None:
        fig_comp.add_trace(go.Bar(y=imp_h['Factor'], x=imp_h['Peso'], name='Hombres', orientation='h', marker_color='#3d5a80'))
    if imp_o is not None:
        fig_comp.add_trace(go.Bar(y=imp_o['Factor'], x=imp_o['Peso'], name='Otros', orientation='h', marker_color='#98c1d9'))
    
    fig_comp.update_layout(title="<b>4. Comparativa de Criterios de Selección (IA)</b>", barmode='group', height=600)
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    tab_m, tab_h, tab_o = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino", "⚧ Perfil Otros"])
    
    def dibujar_radio(datos_gen, resumen, color, pestana, nombre_gen):
        with pestana:
            if resumen is not None:
                df_exito = datos_gen[datos_gen['hiring_decision']==1]
                st.write(f"Análisis de los factores más influyentes para {nombre_gen}:")
                for i, fila in resumen.sort_values('Peso', ascending=False).head(4).iterrows():
                    var = fila['Factor']
                    fig = px.histogram(df_exito, x=var, 
                                       title=f"Factor: {var} (Impacto: {fila['Peso']:.1%})", 
                                       color_discrete_sequence=[color], text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Datos insuficientes para generar perfil de éxito en la categoría {nombre_gen}.")

    dibujar_radio(df[df['gender']=='female'], imp_m, '#e07a5f', tab_m, "Mujeres")
    dibujar_radio(df[df['gender']=='male'], imp_h, '#3d5a80', tab_h, "Hombres")
    dibujar_radio(df[df['gender']=='other'], imp_o, '#98c1d9', tab_o, "Otros")

    # --- SECCIÓN IV: VARIABLES INDEPENDIENTES & PARETO ---
    st.divider()
    st.header("IV. Análisis de Variables Independientes (Pareto Global)")

    # Consolidamos importancia global promediando todos los géneros disponibles
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        imp_global = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        imp_global['Peso_Acumulado'] = (imp_global['Peso'].cumsum() / imp_global['Peso'].sum()) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=imp_global['Factor'], y=imp_global['Peso'], name="Impacto Individual", marker_color='#3d5a80'))
        fig_pareto.add_trace(go.Scatter(x=imp_global['Factor'], y=imp_global['Peso_Acumulado'], name="% Acumulado", yaxis="y2", line=dict(color="#e07a5f", width=3)))

        fig_pareto.update_layout(
            title="<b>5. PARETO: Factores que realmente deciden la contratación</b>",
            yaxis=dict(title="Importancia relativa (IA)"),
            yaxis2=dict(title="Porcentaje Acumulado", overlaying="y", side="right", range=[0, 110]),
            height=500
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Análisis de Doble Estándar en Variables Independientes
        st.subheader("Diferencias en Competencias del Personal Contratado")
        vars_indep = [v for v in ['international_exp', 'entrepeneur_exp', 'programming_exp', 'add_languages'] if v in df.columns]
        
        if vars_indep:
            df_comp_indep = df_solo_contratados.groupby('gender')[vars_indep].mean().reset_index()
            df_comp_melted = df_comp_indep.melt(id_vars='gender', var_name='Variable', value_name='Promedio')

            fig_indep = px.bar(df_comp_melted, x='Variable', y='Promedio', color='gender', barmode='group',
                               title="<b>6. Comparativa de Requisitos Técnicos entre Contratados</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}, 
                               text_auto='.2f')
            st.plotly_chart(fig_indep, use_container_width=True)
            
    st.caption("Nota: Los datos y gráficos presentados cumplen con la nomenclatura NIIF para la presentación de Estados Financieros y reportes de sostenibilidad.")

else:
    st.info("👋 Bienvenida/o. Por favor, cargue su archivo de datos (CSV o Excel) para procesar la auditoría.")
