import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría de Equidad IA", layout="wide")

st.title("📊 Sistema de Auditoría de Equidad")
st.markdown("Análisis de sesgos en la contratación mediante Inteligencia Artificial.")

# --- 1. CARGA Y LIMPIEZA ---
st.sidebar.header("📂 Configuración")
file = st.sidebar.file_uploader("Subir archivo (CSV con ;)", type=['csv', 'xlsx'])

if file:
    # Lectura inteligente
    df = pd.read_csv(file, sep=';') if file.name.endswith('.csv') else pd.read_excel(file)
    df.columns = df.columns.str.lower()
    df['hiring_decision'] = df['hiring_decision'].astype(int)

    features = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                'relevance_of_studies', 'squad']

    # --- 2. DIAGNÓSTICO INICIAL ---
    st.header("I. Diagnóstico de Volumen y Exigencia")
    col1, col2 = st.columns(2)

    with col1:
        df_ok = df[df['hiring_decision'] == 1]
        fig_pie = px.pie(df_ok, names='gender', 
                         title="<b>1. [GRÁFICO DE TORTA] DISTRIBUCIÓN DE CONTRATADOS</b>",
                         color='gender', hole=0.4,
                         color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_box = px.box(df_ok, x='gender', y='score', color='gender',
                         title="<b>2. [GRÁFICO DE CAJA] SCORE REAL DE CONTRATADOS</b>",
                         color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                         points="all")
        st.plotly_chart(fig_box, use_container_width=True)

    # --- 3. PROCESAMIENTO DE MODELOS IA ---
    def entrenar_modelo(data, gen):
        df_gen = data[data['gender'] == gen].copy()
        if df_gen.empty: return None
        X = pd.get_dummies(df_gen[features], drop_first=True)
        y = df_gen['hiring_decision']
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        imp = pd.DataFrame({'Var': X.columns, 'Peso': model.feature_importances_})
        imp['Factor'] = imp['Var'].apply(lambda x: next((f for f in features if x.startswith(f)), x))
        return imp.groupby('Factor')['Peso'].sum().reset_index()

    res_m = entrenar_modelo(df, 'female')
    res_h = entrenar_modelo(df, 'male')

    # --- 4. COMPARATIVA DE CRITERIOS (EL DUELO) ---
    st.divider()
    st.header("II. Análisis de Brechas en la Decisión")
    
    union = pd.merge(res_m, res_h, on='Factor', suffixes=('_Mujeres', '_Hombres')).sort_values('Peso_Mujeres', ascending=True)
    
    fig_union = go.Figure()
    fig_union.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Mujeres'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
    fig_union.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Hombres'], name='Hombres', orientation='h', marker_color='#3d5a80'))
    fig_union.update_layout(title="<b>3. [GRÁFICO DE BARRAS AGRUPADAS] COMPARATIVA DE IMPORTANCIA: Mujeres vs Hombres</b>", 
                          barmode='group', height=500)
    st.plotly_chart(fig_union, use_container_width=True)

    # --- 5. RANKINGS INDIVIDUALES ---
    col3, col4 = st.columns(2)
    
    with col3:
        fig_m = px.bar(res_m.sort_values('Peso', ascending=True), x='Peso', y='Factor', orientation='h',
                       title="<b>4. [GRÁFICO DE BARRAS] RANKING: Mujeres</b>", color_discrete_sequence=['#e07a5f'])
        st.plotly_chart(fig_m, use_container_width=True)

    with col4:
        fig_h = px.bar(res_h.sort_values('Peso', ascending=True), x='Peso', y='Factor', orientation='h',
                       title="<b>5. [GRÁFICO DE BARRAS] RANKING: Hombres</b>", color_discrete_sequence=['#3d5a80'])
        st.plotly_chart(fig_h, use_container_width=True)

    # --- 6. RADIOGRAFÍA DETALLADA ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    
    pestana_m, pestana_h = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino"])

    def dibujar_radiografia(df_orig, resumen, color, tab):
        df_exito = df_orig[df_orig['hiring_decision'] == 1]
        res_ordenado = resumen.sort_values('Peso', ascending=False)
        with tab:
            for i, row in res_ordenado.iterrows():
                var = row['Factor']
                tipo = "HISTOGRAMA" if var in ['age', 'score'] else "GRÁFICO DE BARRAS"
                fig = px.histogram(df_exito, x=var, 
                                   title=f"Prioridad #{i+1}: {var} [{tipo}] (Peso: {row['Peso']:.1%})",
                                   color_discrete_sequence=[color], text_auto=True)
                if var not in ['age', 'score']: fig.update_xaxes(categoryorder="total descending")
                st.plotly_chart(fig, use_container_width=True)

    dibujar_radiografia(df[df['gender']=='female'], res_m, '#e07a5f', pestana_m)
    dibujar_radiografia(df[df['gender']=='male'], res_h, '#3d5a80', pestana_h)

else:
    st.info("👋 Por favor, sube tu archivo CSV o Excel para comenzar el análisis.")
