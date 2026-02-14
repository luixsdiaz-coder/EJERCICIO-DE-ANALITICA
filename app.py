# --- SECCIÓN I: DIAGNÓSTICO DE EQUIDAD ---
    st.header("I. Diagnóstico de Embudo y Sesgos")
    
    # Preparación de datos de conversión
    c_post = df['gender'].value_counts().reset_index()
    c_post.columns = ['gender', 'cantidad']; c_post['estado'] = 'Postulantes'
    
    c_cont = df[df['hiring_decision'] == 1]['gender'].value_counts().reset_index()
    c_cont.columns = ['gender', 'cantidad']; c_cont['estado'] = 'Contratados'
    
    df_conv = pd.concat([c_post, c_cont])

    # Fila 1: El Embudo (Barras)
    st.plotly_chart(px.bar(df_conv, x='gender', y='cantidad', color='estado', barmode='group',
                           title="<b>[GRÁFICO DE BARRAS] Embudo de Selección: ¿Quiénes postulan vs Quiénes entran?</b>",
                           color_discrete_map={'Postulantes': '#9cacaf', 'Contratados': '#3d5a80'},
                           text_auto=True), use_container_width=True)

    # Fila 2: Torta y Cajas
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df[df['hiring_decision']==1], names='gender', hole=0.4, 
                               title="<b>[GRÁFICO DE TORTA] Composición Final del Talento</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df[df['hiring_decision']==1], x='gender', y='score', color='gender', 
                               title="<b>[GRÁFICO DE CAJA] Exigencia de Score (Solo Contratados)</b>",
                               color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'}), use_container_width=True)
