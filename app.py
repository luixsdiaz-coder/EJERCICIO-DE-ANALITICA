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
Esta aplicación utiliza modelos de **Machine Learning (Random Forest)** para detectar si los criterios de contratación 
son los mismos para hombres y mujeres. 
""")

# 2. CARGA DE DATOS
st.sidebar.header("📂 Configuración de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV (separado por ;)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # --- CORRECCIÓN DE LECTURA (Separador ;) ---
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=';')
    else:
        df = pd.read_excel(uploaded_file)
    
    # --- LIMPIEZA DE COLUMNAS (Pasar a minúsculas y quitar espacios) ---
    df.columns = df.columns.str.strip().str.lower()
    
    # Variables a analizar
    features_list = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                    'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                    'relevance_of_studies', 'squad']
    
    # Verificar columnas críticas
    if 'gender' in df.columns and 'hiring_decision' in df.columns:
        
        def calcular_importancia(data, genero):
            # Filtrar por género de forma segura
            df_gen = data[data['gender'].astype(str).str.lower() == genero].copy()
            if len(df_gen) < 10: return None, 0
            
            # Identificar qué variables de nuestra lista existen en el archivo
            columnas_presentes = [f for f in features_list if f in df.columns]
            
            # Convertir variables de texto a números (Variables Dummies)
            X = pd.get_dummies(df_gen[columnas_presentes], drop_first=True)
            y = df_gen['hiring_decision']
            
            # Validar que existan ambos casos (contratados y no contratados)
            if len(y.unique()) < 2: return None, 0
            
            # Entrenamiento del modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            imp_df = pd.DataFrame({'Variable': X.columns, 'Peso': model.feature_importances_})
            return imp_df.sort_values('Peso', ascending=True), acc

        # Ejecutar el modelo para ambos géneros
        imp_mujeres, acc_m = calcular_importancia(df, 'female')
        imp_hombres, acc_h = calcular_importancia(df, 'male')

        # --- SECCIÓN 1: MÉTRICAS DE CONFIABILIDAD ---
        st.subheader("🎯 Confiabilidad del Análisis")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Precisión Modelo Femenino", f"{acc_m:.1%}")
            st.caption("Qué tan predecible es la contratación de mujeres.")
        with col_m2:
            st.metric("Precisión Modelo Masculino", f"{acc_h:.1%}")
            st.caption("Qué tan predecible es la contratación de hombres.")

        # --- SECCIÓN 2: COMPARATIVA DE IMPORTANCIA ---
        st.divider()
        st.subheader("⚖️ ¿Qué pesa más al contratar?")
        st.info("Este gráfico muestra qué tanto influye cada variable en la decisión final de contratar.")
        
        c1, c2 = st.columns(2)
        with c1:
            if imp_mujeres is not None:
                fig_m = px.bar(imp_mujeres, x='Peso', y='Variable', orientation='h', 
                             title="Importancia: Perfil Femenino", color_discrete_sequence=['#e07a5f'])
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.warning("Datos insuficientes para el perfil femenino.")
        
        with c2:
            if imp_hombres is not None:
                fig_h = px.bar(imp_hombres, x='Peso', y='Variable', orientation='h', 
                             title="Importancia: Perfil Masculino", color_discrete_sequence=['#3d5a80'])
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.warning("Datos insuficientes para el perfil masculino.")

        # --- SECCIÓN 3: RADIOGRAFÍA POR VARIABLE ---
        st.divider()
        st.subheader("🔍 Radiografía Detallada")
        st.markdown("Analiza la distribución de las variables en las personas que **sí fueron contratadas**.")
        
        columnas_disponibles = [c for c in features_list if c in df.columns]
        var_seleccionada = st.selectbox("Selecciona una variable para inspeccionar:", columnas_disponibles)
        
        df_contratados = df[df['hiring_decision'] == 1]
        
        fig_dist = px.histogram(df_contratados, 
                               x=var_seleccionada, 
                               color='gender', 
                               barmode='group',
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'},
                               text_auto=True,
                               title=f"Distribución de {var_seleccionada.upper()} en Contratados")
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Conclusión automática en la barra lateral
        st.sidebar.success("✅ Análisis Generado")
        if imp_mujeres is not None:
            top_v = imp_mujeres.sort_values('Peso', ascending=False).iloc[0]['Variable']
            st.sidebar.write(f"Para ellas, lo más importante es: **{top_v}**")

    else:
        st.error(f"Error: No encontré 'gender' o 'hiring_decision'. Columnas detectadas: {list(df.columns)}")

else:
    st.info("👋 Sube tu archivo CSV o Excel en la barra lateral para comenzar.")
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
