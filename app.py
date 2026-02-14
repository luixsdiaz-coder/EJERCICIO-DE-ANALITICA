import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Auditoría Equidad NIIF", layout="wide")
st.title("📊 Auditoría de Equidad en Reclutamiento")

file = st.sidebar.file_uploader("Cargar Dataset", type=['csv', 'xlsx'])

if file:
    df = pd.read_csv(file, sep=';') if file.name.endswith('.csv') else pd.read_excel(file)
    df.columns = df.columns.str.lower()
    df['hiring_decision'] = df['hiring_decision'].astype(int)

    features = ['age', 'nationality', 'sport', 'score', 'degree', 'international_exp', 
                'entrepeneur_exp', 'debateclub', 'programming_exp', 'add_languages', 
                'relevance_of_studies', 'squad']

    # --- SECCIÓN I: DIAGNÓSTICO ---
    st.header("I. Diagnóstico de Sesgos")
    c1, c2 = st.columns(2)
    df_ok = df[df['hiring_decision'] == 1]

    with c1:
        st.plotly_chart(px.pie(df_ok, names='gender', hole=0.4, title="<b>[GRÁFICO DE TORTA] Distribución</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(df_ok, x='gender', y='score', color='gender', title="<b>[GRÁFICO DE CAJA] Exigencia Score</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)

    # --- MODELADO IA ---
    def get_imp(gen):
        d = df[df['gender'] == gen].copy()
        X = pd.get_dummies(d[features], drop_first=True)
        m = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, d['hiring_decision'])
        res = pd.DataFrame({'v': X.columns, 'p': m.feature_importances_})
        res['Factor'] = res['v'].apply(lambda x: next((f for f in features if x.startswith(f)), x))
        return res.groupby('Factor')['p'].sum().reset_index().rename(columns={'p': 'Peso'})

    rm, rh = get_imp('female'), get_imp('male')

    # --- SECCIÓN II: COMPARATIVA ---
    st.divider()
    st.header("II. Comparativa de Importancia")
    union = pd.merge(rm, rh, on='Factor', suffixes=('_Mujeres', '_Hombres')).sort_values('Peso_Mujeres')
    fig_u = go.Figure()
    fig_u.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Mujeres'], name='Mujeres', orientation='h', marker_color='#e07a5f'))
    fig_u.add_trace(go.Bar(y=union['Factor'], x=union['Peso_Hombres'], name='Hombres', orientation='h', marker_color='#3d5a80'))
    st.plotly_chart(fig_u.update_layout(title="<b>[BARRAS AGRUPADAS] Brecha de Criterios</b>", barmode='group'), use_container_width=True)

    # --- SECCIÓN III: RANKINGS ---
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(px.bar(rm.sort_values('Peso'), x='Peso', y='Factor', orientation='h', title="<b>[BARRAS] Ranking Mujeres</b>", color_discrete_sequence=['#e07a5f']), use_container_width=True)
    with c4: st.plotly_chart(px.bar(rh.sort_values('Peso'), x='Peso', y='Factor', orientation='h', title="<b>[BARRAS] Ranking Hombres</b>", color_discrete_sequence=['#3d5a80']), use_container_width=True)

    # --- SECCIÓN IV: RADIOGRAFÍAS ---
    st.divider()
    t1, t2 = st.tabs(["🚺 Perfil Femenino", "🚹 Perfil Masculino"])
    
    def radio(d_gen, res, col, tab):
        with tab:
            for i, r in res.sort_values('Peso', ascending=False).iterrows():
                v = r['Factor']
                tipo = "HISTOGRAMA" if v in ['age', 'score', 'add_languages'] else "BARRAS"
                fig = px.histogram(d_gen[d_gen['hiring_decision']==1], x=v, title=f"#{i+1}: {v} [{tipo}] ({r['Peso']:.1%})", color_discrete_sequence=[col], text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

    radio(df[df['gender']=='female'], rm, '#e07a5f', t1)
    radio(df[df['gender']=='male'], rh, '#3d5a80', t2)
