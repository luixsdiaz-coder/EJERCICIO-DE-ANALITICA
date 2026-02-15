import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = df['hiring_decision'].astype(int)

    variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                      'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO ESTRATÉGICO ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
    df_post = df['gender'].value_counts().reset_index(name='cantidad').assign(estado='Postulantes')
    df_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index(name='cantidad').assign(estado='Contratados')
    df_embudo = pd.concat([df_post, df_cont])

    st.plotly_chart(px.bar(df_embudo, x='gender', y='cantidad', color='estado', barmode='group',
                           title="<b>1. [GRÁFICO DE BARRAS] EMBUDO DE SELECCIÓN</b>",
                           color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'}, text_auto=True), use_container_width=True)

    col1, col2 = st.columns(2)
    df_solo_contratados = df[df['hiring_decision'] == 1]

    with col1:
        st.plotly_chart(px.pie(df_solo_contratados, names='gender', hole=0.4, 
                               title="<b>2. [GRÁFICO DE TORTA] Distribución de Contratados</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}), use_container_width=True)

    with col2:
        st.plotly_chart(px.box(df_solo_contratados, x='gender', y='score', color='gender', 
                              title="<b>3. [GRÁFICO DE CAJA] Nivel de Exigencia (Scores)</b>",
                              color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}), use_container_width=True)

    # --- FUNCIÓN DE IMPORTANCIA IA ---
    def obtener_importancia(gen):
        datos_gen = df[df['gender'] == gen].copy()
        if len(datos_gen) < 5 or datos_gen['hiring_decision'].nunique() < 2: return None
        cols = [c for c in variables_raiz if c in datos_gen.columns]
        X = pd.get_dummies(datos_gen[cols], drop_first=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': model.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in cols if x.startswith(f)), x))
        # Agrupamos y devolvemos sin orden fijo para procesar después
        return res.groupby('Factor')['p'].sum().reset_index(name='Peso')

    imp_m, imp_h, imp_o = obtener_importancia('female'), obtener_importancia('male'), obtener_importancia('other')

    # --- II. IMPORTANCIA COMPARATIVA (ORDENADA DE MAYOR A MENOR) ---
    st.divider()
    st.header("II. Análisis de Importancia de Variables (Orden Mayor a Menor)")
    
    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        # Calculamos el orden global (Promedio de importancia) para que el gráfico se vea de mayor a menor
        orden_global = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=True).reset_index()
        lista_ordenada = orden_global['Factor'].tolist()

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
        
        fig_imp.update_layout(
            title="<b>4. [BARRAS AGRUPADAS] Importancia IA (Ordenado de Mayor Impacto arriba)</b>", 
            barmode='group', height=600,
            yaxis={'categoryorder':'array', 'categoryarray': lista_ordenada}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS INDIVIDUALES CON ESTADÍSTICAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito (Individual)")
    tab_m, tab_h, tab_o = st.tabs(["🚺 Mujeres", "🚹 Hombres", "⚧ Otros"])
    
    def render_perfil(resumen, datos_gen, color, tab):
        with tab:
            if resumen is not None:
                df_exito = datos_gen[datos_gen['hiring_decision']==1]
                # Ordenamos de mayor a menor para los histogramas
                resumen_ordenado = resumen.sort_values('Peso', ascending=False)
                for _, fila in resumen_ordenado.iterrows():
                    var, peso = fila['Factor'], fila['Peso']
                    if var in ['age', 'score', 'add_languages']:
                        v_min, v_max, v_mean = df_exito[var].min(), df_exito[var].max(), df_exito[var].mean()
                        titulo = f"[HISTOGRAMA] {var.upper()} | Media: {v_mean:.2f} | Min: {v_min} | Max: {v_max}"
                        fig = px.histogram(df_exito, x=var, title=titulo, color_discrete_sequence=[color], text_auto=True)
                    else:
                        fig = px.histogram(df_exito, x=var, title=f"[GRÁFICO DE BARRAS] {var.upper()} (Impacto: {peso:.1%})",
                                           color_discrete_sequence=[color], text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Datos insuficientes.")

    render_perfil(imp_m, df[df['gender']=='female'], '#e07a5f', tab_m)
    render_perfil(imp_h, df[df['gender']=='male'], '#3d5a80', tab_h)
    render_perfil(imp_o, df[df['gender']=='other'], '#98c1d9', tab_o)

    # --- SECCIÓN IV: MEZCLA MULTIVARIABLE (ORDENADA) ---
    st.divider()
    st.header("IV. Relación y Mezcla Multivariable (Orden Mayor a Menor)")
    
    if list_imps:
        # Re-calculamos el orden para la mezcla (Descendente)
        imp_desc = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        
        for var in imp_desc['Factor']:
            stats = df_solo_contratados.groupby('gender')[var].agg(['mean', 'min', 'max']).reset_index()
            resumen_stats = " | ".join([f"{r.gender}: μ={r['mean']:.1f}" for r in stats.itertuples()])
            
            fig_mix = px.histogram(df_solo_contratados, x=var, color='gender', barmode='group',
                                   title=f"<b>[BARRAS AGRUPADAS] {var.upper()}</b> <br><sup>{resumen_stats}</sup>",
                                   color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'},
                                   text_auto=True)
            st.plotly_chart(fig_mix, use_container_width=True)

    # --- SECCIÓN V: PARETO ---
    st.divider()
    st.header("V. Análisis de Pareto Global")
    if list_imps:
        imp_desc['Peso_Acum'] = (imp_desc['Peso'].cumsum() / imp_desc['Peso'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=imp_desc['Factor'], y=imp_desc['Peso'], name="Impacto Individual", marker_color='#3d5a80'))
        fig_p.add_trace(go.Scatter(x=imp_desc['Factor'], y=imp_desc['Peso_Acum'], name="% Acumulado", yaxis="y2", line=dict(color="#e07a5f")))
        fig_p.update_layout(title="<b>[PARETO] Jerarquía Final de Decisión</b>", yaxis2=dict(overlaying="y", side="right", range=[0,110]))
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption("Reporte bajo nomenclatura NIIF para la transparencia en Capital Humano.")

else:
    st.info("Suba su archivo para generar el tablero integral de auditoría.")
