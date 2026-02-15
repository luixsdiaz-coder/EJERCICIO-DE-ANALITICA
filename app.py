import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría Integral Equidad NIIF", layout="wide")
st.title("📊 Auditoría de Equidad: Análisis Multivariable")

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

    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']

    # --- FUNCIÓN DE IMPORTANCIA IA ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols = [c for c in variables_raiz if c in datos_gen.columns]
        X = pd.get_dummies(datos_gen[cols], drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso').sort_values('Peso', ascending=False)

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    # --- SECCIÓN I: RADIOGRAFÍAS INDIVIDUALES ---
    st.header("I. Radiografías de Perfiles de Éxito (Individual)")
    tab_m, tab_h, tab_o = st.tabs(["🚺 Mujeres", "🚹 Hombres", "⚧ Otros"])
    
    def render_perfil(resumen, datos_gen, color, tab):
        with tab:
            if resumen is not None:
                df_exito = datos_gen[datos_gen['hiring_decision']==1]
                for _, fila in resumen.iterrows():
                    var, peso = fila['Factor'], fila['Peso']
                    tipo = "[HISTOGRAMA]" if var in ['age', 'score', 'add_languages'] else "[GRÁFICO DE BARRAS]"
                    fig = px.histogram(df_exito, x=var, title=f"{tipo} Variable: {var} (Importancia: {peso:.1%})",
                                       color_discrete_sequence=[color], text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Datos insuficientes.")

    render_perfil(imp_m, df[df['gender']=='female'], '#e07a5f', tab_m)
    render_perfil(imp_h, df[df['gender']=='male'], '#3d5a80', tab_h)
    render_perfil(imp_o, df[df['gender']=='other'], '#98c1d9', tab_o)

    # --- SECCIÓN II: ANÁLISIS MEZCLADO (RELACIÓN DE VARIABLES) ---
    st.divider()
    st.header("II. Relación y Comparativa Unificada (Todos los Géneros)")
    
    df_contratados = df[df['hiring_decision'] == 1]
    
    # Pareto Global para decidir el orden de la mezcla
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        imp_global = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        
        st.write("Variables presentadas en orden de impacto global (Mayor a Menor):")
        for var in imp_global['Factor']:
            fig_mix = px.histogram(df_contratados, x=var, color='gender', barmode='group',
                                   title=f"[GRÁFICO DE BARRAS AGRUPADAS] Relación por Género: {var}",
                                   color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                                   text_auto=True)
            st.plotly_chart(fig_mix, use_container_width=True)

    # --- SECCIÓN III: PARETO GLOBAL ---
    st.divider()
    st.header("III. Pareto Global de la Decisión")
    if list_imps:
        imp_global['Peso_Acum'] = (imp_global['Peso'].cumsum() / imp_global['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_global['Factor'], y=imp_global['Peso'], name="Peso Individual", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_global['Factor'], y=imp_global['Peso_Acum'], name="% Acumulado", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>[PARETO] Jerarquía Final de Contratación</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption("Reporte bajo nomenclatura NIIF para la presentación de Estados Financieros y Diversidad.")

else:
    st.info("Por favor, suba su base de datos para ejecutar la auditoría.")
