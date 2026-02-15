import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from google.colab import files

# --- 1. CARGA DE DATOS ---
print("🚀 Iniciando Auditoría de Equidad NIIF. Por favor, sube tu archivo:")
uploaded = files.upload()
nombre_archivo = list(uploaded.keys())[0]

df = pd.read_csv(nombre_archivo, sep=';') if nombre_archivo.endswith('.csv') else pd.read_excel(nombre_archivo)
df.columns = df.columns.str.lower().str.strip()

# Variables de entrada (Sin incluir el score en el análisis de importancia posterior)
variables_raiz = ['age', 'sport', 'score', 'international_exp', 'entrepeneur_exp', 
                  'debateclub', 'programming_exp', 'add_languages', 'relevance_of_studies', 'squad']

for col in variables_raiz:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'hiring_decision' in df.columns:
    df['hiring_decision'] = pd.to_numeric(df['hiring_decision'], errors='coerce').fillna(0).astype(int)

df_solo_contratados = df[df['hiring_decision'] == 1].copy()
colores_dict = {'female': '#e07a5f', 'male': '#3d5a80', 'other': '#98c1d9'}

# --- I. DIAGNÓSTICO DE EMBUDO Y CONVERSIÓN ---
print("\n" + "="*50 + "\nI. DIAGNÓSTICO DE EMBUDO Y CONVERSIÓN\n" + "="*50)

# 1. Tasa de Conversión
df_funnel = []
for g in df['gender'].unique():
    post = len(df[df['gender'] == g])
    cont = len(df[(df['gender'] == g) & (df['hiring_decision'] == 1)])
    df_funnel.append({'Género': g, 'Etapa': 'Postulantes', 'Cantidad': post})
    df_funnel.append({'Género': g, 'Etapa': 'Contratados', 'Cantidad': cont})
px.funnel(df_funnel, x='Cantidad', y='Etapa', color='Género', title="1. [FUNNEL] Conversión de Selección", color_discrete_map=colores_dict).show()

# 2. Distribución de Puntaje (Boxplot)
px.box(df_solo_contratados, x='gender', y='score', color='gender', title="2. [BOX] Distribución de Puntaje de Contratación", color_discrete_map=colores_dict, points="all").show()

# 3. Porcentaje de Contratados
px.pie(df_solo_contratados, names='gender', hole=0.4, title="3. [PIE] Composición Final de Contratados", color_discrete_map=colores_dict).show()


# --- II. MATRIZ DE CORRELACIÓN ---
print("\n" + "="*50 + "\nII. MATRIZ DE CORRELACIÓN\n" + "="*50)
cols_corr = [c for c in variables_raiz if c in df.columns] + ['hiring_decision']
corr_matrix = df[cols_corr].corr()
px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title="Heatmap: Correlación de Atributos", color_continuous_scale='RdBu_r').show()


# --- III. RADAR DE COMPETENCIAS (PERFIL DE RECLUTAMIENTO) ---
print("\n" + "="*50 + "\nIII. RADAR DE COMPETENCIAS\n" + "="*50)
# Excluimos score para ver perfil de capital humano puro
competencias = ['international_exp', 'programming_exp', 'add_languages', 'entrepeneur_exp', 'relevance_of_studies']
comp_p = [c for c in competencias if c in df_solo_contratados.columns]

if comp_p:
    fig_radar = go.Figure()
    for g in df_solo_contratados['gender'].unique():
        df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
        if not df_g.empty:
            valores = [df_g[c].mean() for c in comp_p] + [df_g[comp_p[0]].mean()]
            fig_radar.add_trace(go.Scatterpolar(r=valores, theta=[c.upper() for c in comp_p + [comp_p[0]]], fill='toself', name=g.capitalize(), line=dict(color=colores_dict.get(g, '#888'))))
    fig_radar.update_layout(title="Perfil de Competencias Promedio (Excluyendo Score)").show()


# --- IV. JERARQUÍA DE IMPORTANCIA (ORDENADO POR IMPORTANCIA GLOBAL) ---
print("\n" + "="*50 + "\nIV. JERARQUÍA DE IMPORTANCIA (IA - SIN SCORE)\n" + "="*50)

# Eliminamos score del análisis de IA porque es una variable dependiente del proceso
vars_ia = [v for v in variables_raiz if v != 'score' and v in df.columns]

def obtener_importancia_sin_score(gen):
    datos_gen = df[df['gender'] == gen].dropna(subset=['hiring_decision']).copy()
    if len(datos_gen) < 10 or datos_gen['hiring_decision'].nunique() < 2: return None
    X = pd.get_dummies(datos_gen[vars_ia].fillna(0), drop_first=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, datos_gen['hiring_decision'])
    res = pd.DataFrame({'Factor': X.columns, 'Peso': model.feature_importances_})
    return res.groupby('Factor')['Peso'].sum().reset_index()

imps = {g: obtener_importancia_sin_score(g) for g in ['female', 'male', 'other']}
list_valid_imps = [v for v in imps.values() if v is not None]

if list_valid_imps:
    # Orden global descendente basado en el promedio de todos los géneros
    global_imp = pd.concat(list_valid_imps).groupby('Factor')['Peso'].mean().sort_values(ascending=False).reset_index()
    fig_imp = go.Figure()
    for g, color in colores_dict.items():
        if imps[g] is not None:
            # Reindexamos para asegurar que todas las barras sigan el mismo orden global
            s = imps[g].set_index('Factor').reindex(global_imp['Factor'].tolist()[::-1]).reset_index().fillna(0)
            fig_imp.add_trace(go.Bar(y=s['Factor'], x=s['Peso'], name=g.capitalize(), orientation='h', marker_color=color))
    fig_imp.update_layout(title="Factores Determinantes Reales (IA sin Score)", barmode='group', height=600).show()


# --- V. ANÁLISIS DE VARIABLES POR GÉNERO (INDEPENDIENTE) ---
print("\n" + "="*50 + "\nV. ANÁLISIS DE VARIABLES POR GÉNERO\n" + "="*50)
for g, color in colores_dict.items():
    if imps[g] is not None:
        print(f"\n" + "-"*30 + f"\n>> ANÁLISIS INDEPENDIENTE: {g.upper()}\n" + "-"*30)
        df_g = df_solo_contratados[df_solo_contratados['gender'] == g]
        # Ordenamos las variables según el peso que la IA le dio específicamente a ESTE género
        vars_ordenadas_g = imps[g].sort_values('Peso', ascending=False)['Factor']
        
        for v in vars_ordenadas_g:
            if v in df_g.columns:
                mu, mn, mx = df_g[v].mean(), df_g[v].min(), df_g[v].max()
                fig_hist = px.histogram(df_g, x=v, 
                                        title=f"Distribución {v.upper()} | μ={mu:.2f} | Rango: [{mn} - {mx}]", 
                                        color_discrete_sequence=[color], text_auto=True)
                fig_hist.show()

print("\nAuditoría finalizada bajo criterios NIIF.")
