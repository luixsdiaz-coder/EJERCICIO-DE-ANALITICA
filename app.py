import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 0. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Auditoría de Equidad NIIF", layout="wide")
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
    
    # Convertir Hiring_decision a entero
    if 'hiring_decision' in df.columns:
        df['hiring_decision'] = df['hiring_decision'].astype(int)

    # Definición de variables raíz
    variables_raiz = ['age', 'sport', 'score', 'international_exp', 
                      'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                      'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO DE EMBUDO Y SESGOS ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
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

    # --- SECCIÓN II: IMPORTANCIA IA (ORDENADA MAYOR A MENOR) ---
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
        
        # Agrupamos y ordenamos descendente para el análisis técnico
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'}).sort_values('Peso', ascending=True)

    imp_m = obtener_importancia('female')
    imp_h = obtener_importancia('male')
    imp_o = obtener_importancia('other')

    if imp_h is not None:
        # Usamos el orden de importancia del grupo mayoritario (hombres) como base comparativa
        orden_factores = imp_h.sort_values('Peso', ascending=True)['Factor'].tolist()
        
        fig_comp = go.Figure()
        
        if imp_m is not None:
            imp_m_sorted = imp_m.set_index('Factor').reindex(orden_factores).reset_index().fillna(0)
            fig_comp.add_trace(go.Bar(y=imp_m_sorted['Factor'], x=imp_m_sorted['Peso'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
            
        fig_comp.add_trace(go.Bar(y=imp_h['Factor'], x=imp_h['Peso'], name='Hombres', orientation='h', marker_color='#3d5a80'))
        
        if imp_o is not None:
            imp_o_sorted = imp_o.set_index('Factor').reindex(orden_factores).reset_index().fillna(0)
            fig_comp.add_trace(go.Bar(y=imp_o_sorted['Factor'], x=imp_o_sorted['Peso'], name='Otros', orientation='h', marker_color='#98c1d9'))
        
        fig_comp.update_layout(
            title="<b>4. Factores de Decisión Ordenados por Importancia (IA)</b>",
            barmode='group', height=600,
            xaxis_title="Peso de la Variable",
            yaxis={'categoryorder':'array', 'categoryarray': orden_factores}
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- SECCIÓN III: RADIOGRAFÍAS ---
    st.divider()
    st.header("III. Radiografía de Perfiles de Éxito")
    tab_m, tab_h, tab_o = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino", "⚧ Perfil Otros"])
    
    def dibujar_radio(datos_gen, resumen, color, pestana, nombre_gen):
        with pestana:
            if resumen is not None:
                df_exito = datos_gen[datos_gen['hiring_decision']==1]
                # Mostramos el top 4 de mayor a menor importancia real
                for i, fila in resumen.sort_values('Peso', ascending=False).head(4).iterrows():
                    var = fila['Factor']
                    fig = px.histogram(df_exito, x=var, 
                                       title=f"Impacto {fila['Peso']:.1%}: {var}", 
                                       color_discrete_sequence=[color], text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Sin datos suficientes para {nombre_gen}.")

    dibujar_radio(df[df['gender']=='female'], imp_m, '#e07a5f', tab_m, "Mujeres")
    dibujar_radio(df[df['gender']=='male'], imp_h, '#3d5a80', tab_h, "Hombres")
    dibujar_radio(df[df['gender']=='other'], imp_o, '#98c1d9', tab_o, "Otros")

    # --- SECCIÓN IV: VARIABLES INDEPENDIENTES & PARETO ---
    st.divider()
    st.header("IV. Análisis de Variables Independientes (Pareto Global)")

    list_imps = [i for i in [imp_m, imp_h, imp_o] if i is not None]
    if list_imps:
        imp_global = pd.concat(list_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
        imp_global['Peso_Acumulado'] = (imp_global['Peso'].cumsum() / imp_global['Peso'].sum()) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=imp_global['Factor'], y=imp_global['Peso'], name="Impacto", marker_color='#3d5a80'))
        fig_pareto.add_trace(go.Scatter(x=imp_global['Factor'], y=imp_global['Peso_Acumulado'], name="% Acum", yaxis="y2", line=dict(color="#e07a5f", width=3)))

        fig_pareto.update_layout(
            title="<b>5. PARETO: Jerarquía de Variables en el Proceso Global</b>",
            yaxis=dict(title="Importancia relativa"),
            yaxis2=dict(title="Acumulado %", overlaying="y", side="right", range=[0, 110]),
            height=500
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.subheader("Diferencias en Perfiles Técnicos")
        vars_indep = [v for v in ['international_exp', 'entrepeneur_exp', 'programming_exp', 'add_languages'] if v in df.columns]
        
        if vars_indep:
            df_comp_indep = df_solo_contratados.groupby('gender')[vars_indep].mean().reset_index()
            df_comp_melted = df_comp_indep.melt(id_vars='gender', var_name='Variable', value_name='Promedio')

            fig_indep = px.bar(df_comp_melted, x='Variable', y='Promedio', color='gender', barmode='group',
                               title="<b>6. Comparativa de Competencias (Doble Estándar)</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}, 
                               text_auto='.2f')
            st.plotly_chart(fig_indep, use_container_width=True)
            
    st.caption("Nota: Reporte generado bajo estándares NIIF para la transparencia en la gestión del talento.")

else:
    st.info("Cargue el archivo CSV/Excel para procesar la Auditoría de Equidad.")
