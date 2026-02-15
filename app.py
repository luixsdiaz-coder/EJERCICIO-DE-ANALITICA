# --- SECCIÓN II: CRITERIOS DE SELECCIÓN (MAPA DE CALOR Y CORRELACIÓN DIRECTA) ---
    st.divider()
    st.header("II. Criterios de Selección y Enfoque de Contratación")
    
    cols_corr = [c for c in variables_raiz if c in df.columns]
    full_cols = cols_corr + ['hiring_decision']
    matriz = df_filtrado[full_cols].corr()
    
    # Escala: -1: Celeste, 0: Amarillo, 1: Verde
    custom_colorscale = [
        [0.0, "rgb(173, 216, 230)"],  # -1
        [0.5, "rgb(255, 255, 0)"],    #  0
        [1.0, "rgb(0, 128, 0)"]       # +1
    ]

    col_mapa, col_resumen = st.columns([1.5, 1])

    with col_mapa:
        # Mapa de Calor Estilo Auditoría
        fig_corr = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.12, shared_xaxes=True)
        
        # Bloque principal
        fig_corr.add_trace(go.Heatmap(z=matriz.loc[cols_corr, cols_corr], x=cols_corr, y=cols_corr, 
                                      colorscale=custom_colorscale, zmin=-1, zmax=1, 
                                      text=matriz.loc[cols_corr, cols_corr].round(2), 
                                      texttemplate="%{text}", showscale=False), row=1, col=1)

        # Fila separada: hiring_decision
        fig_corr.add_trace(go.Heatmap(z=[matriz.loc['hiring_decision', cols_corr].values], x=cols_corr, y=['hiring_decision'], 
                                      colorscale=custom_colorscale, zmin=-1, zmax=1, 
                                      text=[matriz.loc['hiring_decision', cols_corr].round(2).values], 
                                      texttemplate="%{text}"), row=2, col=1)

        fig_corr.update_layout(height=600, title_text="<b>4. Mapa de Calor Completo (Interrelaciones)</b>", template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_resumen:
        # Gráfico Resumido: Correlación Directa con Hiring Decision
        # Extraemos la fila de hiring_decision y quitamos la autocorrelación (1.0)
        correlacion_final = matriz['hiring_decision'].drop('hiring_decision').sort_values(ascending=True)
        
        fig_directa = px.bar(
            x=correlacion_final.values, 
            y=correlacion_final.index, 
            orientation='h',
            title="<b>5. ¿Qué impulsa la Contratación? (Correlación Directa)</b>",
            color=correlacion_final.values,
            color_continuous_scale=custom_colorscale,
            range_color=[-1, 1],
            labels={'x': 'Nivel de Influencia', 'y': 'Variable'}
        )
        
        fig_directa.update_layout(height=600, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_directa, use_container_width=True)
