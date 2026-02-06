import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema Geológico Pronóstico", layout="wide")

# --- 2. LOGICA DE COLORES ---
MAPEO_LITOLOGICO = {
    (253,192,122): ['OS', 'OI', 'Oligoceno Superior', 'Oligoceno Inferior'],
    (253, 180, 108): ['ES', 'Eoceno Superior', 'EM','Eoceno Medio', 'EI', 'Eoceno Inferior'],
    (253, 167, 95): ['PS', 'PI', 'Paleoceno Superior', 'Paleoceno Inferior'],
    (166, 216, 74): ['KS', 'KS Mendez', 'Mendez', 'KS San Felipe', 'San Felipe', 'KS Agua Nueva', 'Agua Nueva'],
    (148, 210, 80): ['KM', 'Tamabra', 'El Abra', 'Tamaulipas Superior'],
    (145, 205, 87): ['KI', 'KI Tamaulipas Inferior', 'Tamaulipas Inferior', 'Otates'],
    (217, 241, 247): ['JST', 'Titho', 'Tithoniano'],
    (204, 236, 244): ['JSK', 'Kimmer', 'Kimmeridgiano'],
    (191, 231, 241): ['JSO', 'Oxford', 'Oxfordiano'],
}

def obtener_color(nombre_cima):
    nombre_clean = str(nombre_cima).lower().strip()
    for rgb, palabras in MAPEO_LITOLOGICO.items():
        color_normalizado = tuple(c / 255 for c in rgb)
        for p in palabras:
            if p.lower() in nombre_clean: return color_normalizado
    return (0.8, 0.8, 0.8) 

# --- 3. FUNCION DE CALCULO ---
def calcular_predicciones(df_a, df_b, delta_t, delta_vavg, delta_err, tvdss_b0):
    df_a = df_a.copy()
    df_b = df_b.copy()
    
    # Cálculos Pozo A
    df_a['TVT_Real'] = (df_a['TVDSS (m)'].diff(-1) * -1).fillna(0)
    dtwt_a = (df_a['TWT (s)'].diff(-1) * -1)
    df_a['TVT_Sismico'] = (dtwt_a * df_a['Vel avg (m/s)'] / 2).fillna(0)
    df_a['Err_Rel_Porcentual'] = 0.0
    mask_a = (df_a['Surf'] == 'Concordante') & (df_a['TVT_Real'] > 0)
    df_a.loc[mask_a, 'Err_Rel_Porcentual'] = ((df_a['TVT_Real'] - df_a['TVT_Sismico']) / df_a['TVT_Real']) * 100
    dict_errores = df_a.set_index('Cima')['Err_Rel_Porcentual'].to_dict()
    
    # Cálculos Pozo B
    results = df_b.copy()
    results['dtwt'] = (results['TWT Pred (s)'].diff(-1) * -1).fillna(0)
    results['err_ref_usado'] = results['Cima'].map(dict_errores).fillna(0.0)
    
    mask_conc = results['Superficie'] == 'Concordante'
    results['esp_base'] = 0.0
    results.loc[mask_conc, 'esp_base'] = (results['dtwt'] * results['Vel avg (m/s)'] / 2)
    results['ajuste_err'] = abs(results['esp_base'] * (results['err_ref_usado'] / 100))
    
    term_t = (delta_t * results['Vel avg (m/s)'] / 2)**2
    term_v = (results['dtwt'] * delta_vavg / 2)**2
    results['err_prop_capa'] = np.sqrt(term_t + term_v + delta_err**2)
    
    results['Prediccion_Base'] = (results['esp_base'].cumsum().shift(1).fillna(0)) + tvdss_b0
    results['Prop_Sigma_Acum'] = np.sqrt((results['err_prop_capa']**2).cumsum().shift(1).fillna(0))
    results['Prop_Max'] = results['Prediccion_Base'] + results['Prop_Sigma_Acum']
    results['Prop_Min'] = results['Prediccion_Base'] - results['Prop_Sigma_Acum']
    results['Esp_Calc_Neg'] = ((results['esp_base'] - results['ajuste_err']).cumsum().shift(1).fillna(0)) + tvdss_b0
    results['Esp_Calc_Pos'] = ((results['esp_base'] + results['ajuste_err']).cumsum().shift(1).fillna(0)) + tvdss_b0
    
    return df_a, results

# --- 4. INTERFAZ ---
st.title("⚒️ Sistema de Pronóstico Geológico Profesional")

tab1, tab2, tab3, tab4 = st.tabs(["📍 Referencia", "🔮 Pronóstico", "📉 Validación Real", "💾 Exportar"])

# Configuración de Columnas con Dropdown
config_superficie = {
    "Surf": st.column_config.SelectboxColumn("Superficie", options=["Concordante", "Discordante"], required=True),
    "Superficie": st.column_config.SelectboxColumn("Superficie", options=["Concordante", "Discordante"], required=True)
}

with tab1:
    c1, c2, c3 = st.columns(3)
    dt_v = c1.number_input("Delta T (s):", value=0.002, format="%.3f")
    dv_v = c2.number_input("Delta V (m/s):", value=50.0)
    de_v = c3.number_input("Delta Err:", value=5.0)
    df_a_in = st.data_editor(pd.DataFrame({
        'Surf': ['Concordante']*4, 'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TVDSS (m)': [1848, 2052, 2076, 2116], 'TWT (s)': [1.305, 1.404, 1.416, 1.434], 'Vel avg (m/s)': [4135, 4260, 4525, 5157]
    }), num_rows="dynamic", column_config=config_superficie, key="e_a")

with tab2:
    c4, c5 = st.columns(2)
    tvdss_b = c4.number_input("TVDSS Inicial Pozo B (m):", value=1927)
    escenario = c5.selectbox("Escenario:", ['todos', 'base', 'somero', 'profundo'])
    df_b_in = st.data_editor(pd.DataFrame({
        'Superficie': ['Concordante']*4, 'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TWT Pred (s)': [1.274, 1.314, 1.314, 1.314], 'Vel avg (m/s)': [4362, 4260, 4525, 5157]
    }), num_rows="dynamic", column_config=config_superficie, key="e_b")

with tab3:
    show_r = st.checkbox("Graficar Pozo Real")
    df_r_in = st.data_editor(pd.DataFrame(columns=['Surf', 'Cima', 'TWT Real (s)', 'TVDSS Real (m)']), 
                             num_rows="dynamic", column_config=config_superficie, key="e_r")

with tab4:
    st.subheader("Descargar Resultados")
    nombre_archivo = st.text_input("Nombre del archivo Excel:", "Resultados_Pronostico.xlsx")

# --- 5. PROCESAMIENTO Y GRÁFICO ---
if st.button("📊 EJECUTAR ANÁLISIS", use_container_width=True, type="primary"):
    df_a_res, res_b = calcular_predicciones(df_a_in, df_b_in, dt_v, dv_v, de_v, tvdss_b)
    
    # Gráfico (Mismo código de visualización)
    fig, (ax_well, ax_main) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 4]})
    y_max = res_b['Prop_Max'].max() + 50
    y_min = res_b['Prediccion_Base'].min() - 50

    for i in range(len(res_b)):
        y_t, y_b = res_b['Prediccion_Base'].iloc[i], (res_b['Prediccion_Base'].iloc[i+1] if i+1 < len(res_b) else y_max)
        ax_well.axhspan(y_t, y_b, facecolor=obtener_color(res_b['Cima'].iloc[i]), alpha=0.7, edgecolor='black')
    
    ax_well.set_ylim(y_max, y_min)
    ax_main.plot(res_b['TWT Pred (s)'], res_b['Prediccion_Base'], 'g-o', label='Pronóstico')
    ax_main.set_ylim(y_max, y_min)
    ax_main.legend()
    st.pyplot(fig)

    # --- LÓGICA DE EXPORTACIÓN REAL ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        res_b.to_excel(writer, index=False, sheet_name='Pronostico')
        df_a_res.to_excel(writer, index=False, sheet_name='Referencia_Analizada')
    
    st.download_button(
        label="📥 DESCARGAR EXCEL",
        data=output.getvalue(),
        file_name=nombre_archivo,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
