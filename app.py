import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Auditoría de Equidad", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento")

archivo = st.sidebar.file_uploader("Cargar Base de Datos", type=['csv', 'xlsx'])

if archivo:
    df = pd.read_csv(archivo, sep=';') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.str.lower()
    df['hiring_decision'] = df['hiring_decision'].astype(int)

    variables_raiz = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                      'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                      'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO ---
    st.header("I. Diagnóstico de Sesgos")
    col1, col2 = st.columns(2)
    df_contratados = df[df['hiring_decision'] == 1]

    with col1:
        st.plotly_chart(px.pie(df_contratados, names='gender', hole=0.4, title="<b>[GRÁFICO DE TORTA] Distribución de Contratados</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df_contratados, x='gender', y='score', color='gender', title="<b>[GRÁFICO DE CAJA] Exigencia de Puntaje (Score)</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)

    # --- MODELADO ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        X = pd.get_dummies(datos_gen[variables_raiz], drop_first=True)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': modelo.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in variables_raiz if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'})

    importancia_m, importancia_h = obtener_importancia('female'), obtener_importancia('male')

    # --- SECCIÓN II: COMPARATIVA ---
    st.divider()
    st.header("II. Comparativa de Criterios de Selección")
    union = pd.merge(importancia_m, importancia_h, on='Factor', suffixes=('_Mujeres', '_Hombres')).sort_values('Peso_Mujeres')
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Mujeres'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
    fig_comp.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Hombres'], name='Hombres', orientation='h', marker_color='#3d5a80'))
    st.plotly_chart(fig_comp.update_layout(title="<b>[BARRAS AGRUPADAS] Brecha de Criterios IA</b>", barmode='group'), use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    tab_m, tab_h = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino"])
    
    def dibujar_radio(datos_gen, resumen, color, pestana, titulo_gen):
        with pestana:
            for i, fila in resumen.sort_values('Peso', ascending=False).iterrows():
                var = fila['Factor']
                tipo = "HISTOGRAMA" if var in ['age', 'score', 'add_languages'] else "BARRAS"
                fig = px.histogram(datos_gen[datos_gen['hiring_decision']==1], x=var, 
                                   title=f"#{i+1}: {var} [{tipo}] (Impacto: {fila['Peso']:.1%})", 
                                   color_discrete_sequence=[color], text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    dibujar_radio(df[df['gender']=='female'], importancia_m, '#e07a5f', tab_m, "Mujer")
    dibujar_radio(df[df['gender']=='male'], importancia_h, '#3d5a80', tab_h, "Hombre")
