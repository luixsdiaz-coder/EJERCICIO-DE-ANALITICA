import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría Integral Equidad NIIF", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento")

# --- 1. CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Cargar Base de Datos (CSV o Excel)", type=['csv', 'xlsx'])

if archivo:
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)
    
    df.columns = df.columns.str.lower().str.strip()
    
    # Limpieza y Conversión Numérica
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']
    
    for col in variables_raiz:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = pd.to_numeric(df['hiring_decision'], errors='coerce').fillna(0).astype(int)

    # --- SECCIÓN I: DIAGNÓSTICO ESTRATÉGICO ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
    df_post = df['gender'].value_counts().reset_index(name='cantidad').assign(estado='Postulantes')
    df_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index(name='cantidad').assign(estado='Contratados')
    df_embudo = pd.concat([df_post, df_cont])

    st.plotly_chart(px.bar(df_embudo, x='gender', y='cantidad', color='estado', barmode='group',
                           title="<b>1. [GRÁFICO DE BARRAS] EMBUDO DE SELECCIÓN</b>",
                           color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'}, text_auto=True), use_container_width=True)

    col1, col2 = st.columns(2)
    df_solo_contratados = df[df['hiring_decision'] == 1].copy()

    with col1:
        st.plotly_chart(px.pie(df_solo_contratados, names='gender', hole=0.4, 
                               title="<b>2. [GRÁFICO DE TORTA] Distribución de Contratados</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}), use_container_width=True)

    with col2:
        st.plotly_chart(px.box(df_solo_contratados, x='gender', y='score', color='gender', 
                              title="<b>3. [GRÁFICO DE CAJA] Nivel de Exigencia (Scores)</b>",
                              color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}), use_container_width=True)

    # --- FUNCIÓN IA ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols_presentes = [c for c in variables_raiz if c in datos_gen.columns]
        X = datos_gen[cols_presentes].fillna(0)
        X = pd.get_dummies(X, drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols_presentes if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso')

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    # --- SECCIÓN II: CAMPANAS DE GAUSS (NUEVA) ---
    st.divider()
    st.header("II. Análisis de Distribución Normal (Campanas de Gauss)")
    st.write("Visualización de la normalidad en el perfil de los contratados para variables críticas.")

    vars_gauss = ['age', 'add_languages', 'international_exp', 'programming_exp']
    
    for var in vars_gauss:
        if var in df_solo_contratados.columns:
            fig_gauss = go.Figure()
            colores = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}
            
            for g in df_solo_contratados['gender'].unique():
                data_g = df_solo_contratados[df_solo_contratados['gender'] == g][var].dropna()
                if len(data_g) > 1:
                    mu, std = data_g.mean(), data_g.std()
                    x = np.linspace(data_g.min(), data_g.max(), 100)
                    y = norm.pdf(x, mu, std)
                    
                    # Histograma de base
                    fig_gauss.add_trace(go.Histogram(x=data_g, nbinsx=20, name=f"Datos {g}", 
                                                     marker_color=colores.get(g, '#888'), opacity=0.3, histnorm='probability density'))
                    # Campana de Gauss
                    fig_gauss.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Gauss {g} (μ={mu:.1f})", 
                                                   line=dict(color=colores.get(g, '#888'), width=3)))

            fig_gauss.update_layout(title=f"<b>[CAMPANA DE GAUSS] Distribución de {var.upper()}</b>",
                                    xaxis_title=var.capitalize(), yaxis_title="Densidad de Probabilidad",
                                    height=450, barmode='overlay')
            st.plotly_chart(fig_gauss, use_container_width=True)

    # --- SECCIÓN III: IMPORTANCIA IA ---
    st.divider()
    st.header("III. Importancia de Variables (Orden Mayor a Menor)")
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        orden_desc = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=True).reset_index()
        lista_ordenada = orden_desc['Factor'].tolist()
        fig_imp = go.Figure()
        if imp_m is not None:
            m_s = imp_m.set_index('Factor').reindex(lista_ordenada).reset_index().fillna(0)
            fig_imp.add_trace(go.Bar(y=m_s['Factor'], x=m_s['Peso'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
        if imp_h is not None:
            h_s = imp_h.set_index('Factor').reindex(lista_ordenada).reset_index().fillna(0)
            fig_imp.add_trace(go.Bar(y=h_s['Factor'], x=h_s['Peso'], name='Hombres', orientation='h', marker_color='#3d5a80'))
        if imp_o is not None:
            o_s = imp_o.set_index('Factor').reindex(lista_ordenada).reset_index().fillna(0)
            fig_imp.add_trace(go.Bar(y=o_s['Factor'], x=o_s['Peso'], name='Otros', orientation='h', marker_color='#98c1d9'))
        fig_imp.update_layout(title="<b>[BARRAS AGRUPADAS] Jerarquía Técnica IA</b>", barmode='group', height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- SECCIÓN IV: RADIOGRAFÍAS INDIVIDUALES ---
    st.divider()
    st.header("IV. Radiografía de Perfiles de Éxito (Individual)")
    tabs = st.tabs(["🚺 Mujeres", "🚹 Hombres", "⚧ Otros"])
    for i, g in enumerate(['female', 'male', 'other']):
        with tabs[i]:
            res = [imp_m, imp_h, imp_o][i]
            if res is not None:
                df_g = df[(df['gender'] == g) & (df['hiring_decision'] == 1)]
                res_ord = res.sort_values('Peso', ascending=False)
                for _, f in res_ord.iterrows():
                    v = f['Factor']
                    if pd.api.types.is_numeric_dtype(df_g[v]):
                        titulo = f"[HISTOGRAMA] {v.upper()} | μ={df_g[v].mean():.1f} | Min={df_g[v].min()} | Max={df_g[v].max()}"
                    else: titulo = f"[BARRAS] {v.upper()}"
                    st.plotly_chart(px.histogram(df_g, x=v, title=titulo, color_discrete_sequence=[['#e07a5f', '#3d5a80', '#98c1d9'][i]], text_auto=True), use_container_width=True)

    # --- SECCIÓN V: MEZCLA Y PARETO ---
    st.divider()
    st.header("V. Mezcla Multivariable y Pareto Global")
    if list_imps:
        imp_glob = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        for v in imp_glob['Factor']:
            stats = df_solo_contratados.groupby('gender')[v].agg(['mean']).reset_index()
            res_stats = " | ".join([f"{r.gender}: μ={getattr(r, 'mean'):.1f}" for r in stats.itertuples()])
            st.plotly_chart(px.histogram(df_solo_contratados, x=v, color='gender', barmode='group', title=f"<b>[MEZCLA] {v.upper()}</b> <br><sup>{res_stats}</sup>", color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}, text_auto=True), use_container_width=True)
        
        imp_glob['Peso_Acum'] = (imp_glob['Peso'].cumsum() / imp_glob['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_glob['Factor'], y=imp_glob['Peso'], name="Peso", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_glob['Factor'], y=imp_glob['Peso_Acum'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>[PARETO] Jerarquía Final</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption("Reporte bajo nomenclatura NIIF para la transparencia en Capital Humano.")

else:
    st.info("Suba su archivo para iniciar la auditoría.")
