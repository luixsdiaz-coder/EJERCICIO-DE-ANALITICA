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
    df.columns = df.columns.str.lower()
    
    # Convertir Hiring_decision a entero (1 y 0)
    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = df['hiring_decision'].astype(int)

    variables_raiz = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                      'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                      'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO DE EMBUDO Y SESGOS ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
    # Preparación de datos para el Gráfico de Embudo (Postulantes vs Contratados)
    conteo_post = df['gender'].value_counts().reset_index()
    conteo_post.columns = ['genero', 'cantidad']
    conteo_post['estado'] = 'Postulantes'
    
    conteo_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index()
    conteo_cont.columns = ['genero', 'cantidad']
    conteo_cont['estado'] = 'Contratados'
    
    df_embudo = pd.concat([conteo_post, conteo_cont])

    # Gráfico de Barras: Comparativa de conversión
    fig_embudo = px.bar(df_embudo, x='genero', y='cantidad', color='estado', barmode='group',
                        title="<b>1. [GRÁFICO DE BARRAS] EMBUDO DE SELECCIÓN: Postulantes vs Contratados</b>",
                        color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'},
                        text_auto=True)
    st.plotly_chart(fig_embudo, use_container_width=True)

    col1, col2 = st.columns(2)
    df_solo_contratados = df[df['hiring_decision'] == 1]

    with col1:
        # Gráfico de Torta: Composición de los que entraron
        fig_torta = px.pie(df_solo_contratados, names='gender', hole=0.4, 
                           title="<b>2. [GRÁFICO DE TORTA] Distribución Final de Contratados</b>",
                           color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'})
        st.plotly_chart(fig_torta, use_container_width=True)

    with col2:
        # Gráfico de Caja: Exigencia de Score
        fig_caja = px.box(df_solo_contratados, x='gender', y='score', color='gender', 
                          title="<b>3. [GRÁFICO DE CAJA] Nivel de Exigencia (Solo Contratados)</b>",
                          color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                          points="all")
        st.plotly_chart(fig_caja, use_container_width=True)

    # --- SECCIÓN II: IMPORTANCIA IA ---
    st.divider()
    st.header("II. Análisis de Importancia de Variables")

    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if datos_gen.empty: return None
        X = pd.get_dummies(datos_gen[variables_raiz], drop_first=True)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': modelo.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in variables_raiz if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'})

    imp_m = obtener_importancia('female')
    imp_h = obtener_importancia('male')

    # Gráfico Comparativo Agrupado
    union = pd.merge(imp_m, imp_h, on='Factor', suffixes=('_Mujeres', '_Hombres')).sort_values('Peso_Mujeres')
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Mujeres'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
    fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Hombres'], name='Hombres', orientation='h', marker_color='#3d5a80'))
    fig_comp.update_layout(title="<b>4. [GRÁFICO DE BARRAS AGRUPADAS] Comparativa de Criterios (IA)</b>", barmode='group', height=500)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Rankings Individuales
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(imp_m.sort_values('Peso'), x='Peso', y='Factor', orientation='h', 
                               title="<b>5. [GRÁFICO DE BARRAS] Ranking: Mujeres</b>", color_discrete_sequence=['#e07a5f']), use_container_width=True)
    with c4:
        st.plotly_chart(px.bar(imp_h.sort_values('Peso'), x='Peso', y='Factor', orientation='h', 
                               title="<b>6. [GRÁFICO DE BARRAS] Ranking: Hombres</b>", color_discrete_sequence=['#3d5a80']), use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    tab_m, tab_h = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino"])
    
    def dibujar_radio(datos_gen, resumen, color, pestana):
        with pestana:
            df_exito = datos_gen[datos_gen['hiring_decision']==1]
            for i, fila in resumen.sort_values('Peso', ascending=False).iterrows():
                var = fila['Factor']
                tipo = "HISTOGRAMA" if var in ['age', 'score', 'add_languages'] else "GRÁFICO DE BARRAS"
                fig = px.histogram(df_exito, x=var, 
                                   title=f"#{i+1}: {var} [{tipo}] (Impacto: {fila['Peso']:.1%})", 
                                   color_discrete_sequence=[color], text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    dibujar_radio(df[df['gender']=='female'], imp_m, '#e07a5f', tab_m)
    dibujar_radio(df[df['gender']=='male'], imp_h, '#3d5a80', tab_h)

else:
    st.info("Suba un archivo para iniciar el análisis.")
