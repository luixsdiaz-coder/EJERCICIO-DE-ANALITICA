import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Auditoría IA: Equidad de Género", layout="wide")

st.title("📊 Auditoría de Equidad con Inteligencia Artificial")

# CARGA DE DATOS
st.sidebar.header("📂 Configuración de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Leer el archivo
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # --- CORRECCIÓN DE COLUMNAS (Para evitar el KeyError) ---
    # Esto pone todos los nombres en minúsculas y quita espacios vacíos
    df.columns = df.columns.str.strip().str.lower()
    
    # Definimos las variables buscando sus nombres en minúsculas
    features = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                'relevance_of_studies', 'squad']
    
    # Verificar si existen las columnas críticas
    if 'gender' in df.columns and 'hiring_decision' in df.columns:
        
        def calcular_importancia(data, genero):
            df_gen = data[data['gender'].str.lower() == genero].copy()
            if len(df_gen) < 5: return None, 0
            
            # Solo usamos las columnas que sí existan en tu Excel
            columnas_presentes = [f for f in features if f in df.columns]
            X = pd.get_dummies(df_gen[columnas_presentes], drop_first=True)
            y = df_gen['hiring_decision']
            
            if len(y.unique()) < 2: return None, 0 # Evita error si nadie fue contratado
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            imp_df = pd.DataFrame({'Var': X.columns, 'Peso': model.feature_importances_})
            return imp_df, acc

        # Ejecutar
        imp_mujeres, acc_m = calcular_importancia(df, 'female')
        imp_hombres, acc_h = calcular_importancia(df, 'male')

        # MOSTRAR MÉTRICAS
        st.subheader("Resultados de Confiabilidad")
        c1, c2 = st.columns(2)
        c1.metric("Precisión (Modelo Mujeres)", f"{acc_m:.1%}")
        c2.metric("Precisión (Modelo Hombres)", f"{acc_h:.1%}")

        # GRÁFICO DE IMPORTANCIA
        if imp_mujeres is not None:
            st.divider()
            st.subheader("Variables que más influyen en la contratación")
            fig = px.bar(imp_mujeres.sort_values('Peso'), x='Peso', y='Var', orientation='h', title="Perfil Femenino")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Error: No encontré las columnas 'gender' o 'hiring_decision'. Tus columnas actuales son: {list(df.columns)}")

else:
    st.info("👋 Sube tu archivo en la barra lateral para iniciar.")
