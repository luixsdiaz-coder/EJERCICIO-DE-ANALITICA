import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Auditoría IA: Equidad de Género", layout="wide")

st.title("📊 Auditoría de Equidad con Inteligencia Artificial")
st.markdown("""
Esta aplicación analiza los sesgos de contratación utilizando modelos de **Random Forest**. 
Sube tu archivo para identificar qué variables (Score, Edad, etc.) están frenando la equidad.
""")

# 2. CARGA DE DATOS
st.sidebar.header("📂 Configuración de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Leer el archivo según el formato
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 3. LÓGICA DE PROCESAMIENTO E IA
    features = ['age', 'nationality', 'sport', 'Score', 'Degree', 'International_exp', 
                'Entrepeneur_exp', 'Debateclub', 'Programming_exp', 'Add_languages', 
                'Relevance_of_studies', 'Squad']

    def calcular_importancia(data, genero):
        df_gen = data[data['gender'] == genero].copy()
        if len(df_gen) < 10: return None, 0 # Validación mínima de datos
        
        X = pd.get_dummies(df_gen[features], drop_first=True)
        y = df_gen['Hiring_decision']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        imp_df = pd.DataFrame({'Var': X.columns, 'Peso': model.feature_importances_})
        imp_df['Factor'] = imp_df['Var'].apply(lambda x: next((f for f in features if x.startswith(f)), x))
        resumen = imp_df.groupby('Factor')['Peso'].sum().reset_index()
        return resumen, acc

    # Ejecutar cálculos
    imp_mujeres, acc_m = calcular_importancia(df, 'female')
    imp_hombres, acc_h = calcular_importancia(df, 'male')

    # 4. DASHBOARD - MÉTRICAS PRINCIPALES
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confiabilidad (Mujeres)", f"{acc_m:.1%}")
    with col2:
        st.metric("Confiabilidad (Hombres)", f"{acc_h:.1%}")
    with col3:
        diff_score = df[df['gender']=='female']['Score'].mean() - df[df['gender']=='male']['Score'].mean()
        st.metric("Brecha de Score Promedio", f"{diff_score:.2f} pts")

    st.divider()

    # 5. GRÁFICO COMPARATIVO DE IMPORTANCIA
    st.subheader("⚖️ Comparativa de Criterios: ¿Qué pesa más para cada género?")
    if imp_mujeres is not None and imp_hombres is not None:
        comparativa = pd.merge(imp_mujeres, imp_hombres, on='Factor', suffixes=('_Mujeres', '_Hombres'))
        comparativa = comparativa.sort_values(by='Peso_Mujeres', ascending=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(y=comparativa['Factor'], x=comparativa['Peso_Mujeres'], 
                                 name='Mujeres', orientation='h', marker_color='#e07a5f'))
        fig_comp.add_trace(go.Bar(y=comparativa['Factor'], x=comparativa['Peso_Hombres'], 
                                 name='Hombres', orientation='h', marker_color='#3d5a80'))
        fig_comp.update_layout(barmode='group', height=500)
        st.plotly_chart(fig_comp, use_container_width=True)

    # 6. ANÁLISIS POR VARIABLE
    st.divider()
    st.subheader("🔍 Radiografía por Variable")
    var_analizar = st.selectbox("Selecciona una variable para ver su distribución en contratados", features)
    
    df_contratados = df[df['Hiring_decision'] == True]
    fig_dist = px.histogram(df_contratados, x=var_analizar, color='gender', barmode='group',
                           color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'},
                           text_auto=True, title=f"Distribución de {var_analizar} en personal contratado")
    st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("👋 Por favor, sube un archivo CSV o Excel desde la barra lateral para comenzar el análisis.")
