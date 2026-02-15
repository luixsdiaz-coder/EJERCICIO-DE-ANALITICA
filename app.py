# --- SECCIÓN IV: ANÁLISIS DE VARIABLES INDEPENDIENTES & PARETO ---
    st.divider()
    st.header("IV. Análisis de Variables Independientes (Pareto)")

    # 1. Preparación de datos para Pareto Global
    # Promediamos la importancia de IA de ambos géneros para obtener el peso real de la variable
    imp_global = pd.concat([imp_m, imp_h]).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
    imp_global['Peso_Acumulado'] = (imp_global['Peso'].cumsum() / imp_global['Peso'].sum()) * 100

    # Gráfico de Pareto
    fig_pareto = go.Figure()

    # Barras: Impacto Individual
    fig_pareto.add_trace(go.Bar(
        x=imp_global['Factor'], 
        y=imp_global['Peso'],
        name="Impacto Individual", 
        marker_color='#3d5a80'
    ))

    # Línea: Porcentaje Acumulado
    fig_pareto.add_trace(go.Scatter(
        x=imp_global['Factor'], 
        y=imp_global['Peso_Acumulado'],
        name="% Acumulado", 
        yaxis="y2", 
        line=dict(color="#e07a5f", width=3), 
        marker=dict(size=8)
    ))

    fig_pareto.update_layout(
        title="<b>7. [PARETO] Influencia de Variables Independientes en la Contratación</b>",
        yaxis=dict(title="Peso de la Variable (Importancia IA)"),
        yaxis2=dict(title="Porcentaje Acumulado (%)", overlaying="y", side="right", range=[0, 110]),
        legend=dict(x=0.8, y=1.1),
        height=500
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    # 2. Comparativa de Variables Independientes Clave (Doble Estándar)
    st.subheader("Análisis Comparativo de Competencias")
    
    vars_criticas = ['international_exp', 'entrepeneur_exp', 'programming_exp', 'add_languages']
    
    # Resumen de promedios por género solo para los contratados
    df_comp_indep = df_solo_contratados.groupby('gender')[vars_criticas].mean().reset_index()
    df_comp_melted = df_comp_indep.melt(id_vars='gender', var_name='Variable', value_name='Promedio')

    fig_indep = px.bar(
        df_comp_melted, x='Variable', y='Promedio', color='gender', barmode='group',
        title="<b>8. [BARRAS AGRUPADAS] Perfil Técnico: Mujeres vs Hombres Contratados</b>",
        color_discrete_map={'female': '#e07a5f', 'male': '#3d5a80'},
        text_auto='.2f'
    )
    
    fig_indep.update_layout(yaxis_title="Proporción / Promedio de la Variable")
    st.plotly_chart(fig_indep, use_container_width=True)
