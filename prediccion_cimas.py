import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema Geológico Pronóstico", layout="wide")

# --- 2. CONFIGURACIÓN DE COLORES Y LÓGICA ---
MAPEO_LITOLOGICO = {
    (253,192,122): ['OS', 'OI', 'Oligoceno Superior', 'Oligoceno Inferior'],
    (253, 180, 108): ['ES', 'Eoceno Superior', 'EM','Eoceno Medio', 'EI', 'Eoceno Inferior'],
    (253, 167, 95): ['PS', 'PI', 'Paleoceno Superior', 'Paleoceno Inferior'],
    (166, 216, 74): ['KS', 'KS Mendez', 'Mendez', 'KS San Felipe', 'San Felipe', 'KS Agua Nueva', 'Agua Nueva', 'KS SF', 'KS AN' ],
    (148, 210, 80): ['KM', 'Tamabra', 'El Abra', 'Tamaulipas Superior'],
    (145, 205, 87): ['KI', 'KI Tamaulipas Inferior', 'Tamaulipas Inferior', 'Otates', 'KTI'],
    (217, 241, 247): ['JST', 'Titho', 'Tithoniano', 'JS Titho'],
    (204, 236, 244): ['JSK', 'Kimmer', 'Kimmeridgiano', 'JS Kimmer'],
    (191, 231, 241): ['JSO', 'Oxford', 'Oxfordiano', 'JS Oxford'],
}

def obtener_color(nombre_cima):
    nombre_clean = str(nombre_cima).lower().strip()
    for rgb, palabras in MAPEO_LITOLOGICO.items():
        color_normalizado = tuple(c / 255 for c in rgb)
        for p in palabras:
            if p.lower() in nombre_clean:
                return color_normalizado
    return (0.8, 0.8, 0.8) 

def calcular_predicciones(df_a, df_b, delta_t, delta_vavg, delta_err):
    df_a = df_a.copy()
    df_b = df_b.copy()
    
    # Lógica de Pozo A
    df_a['TVT_Real'] = (df_a['TVDSS (m)'].diff(-1) * -1).fillna(0)
    dtwt_a = (df_a['TWT (s)'].diff(-1) * -1)
    df_a['TVT_Sismico'] = (dtwt_a * df_a['Vel avg (m/s)'] / 2).fillna(0)
    
    df_a['Err_Rel_Porcentual'] = 0.0
    mask_a = (df_a['Surf'] == 'Concordante') & (df_a['TVT_Real'] > 0)
    df_a.loc[mask_a, 'Err_Rel_Porcentual'] = ((df_a['TVT_Real'] - df_a['TVT_Sismico']) / df_a['TVT_Real']) * 100
    dict_errores = df_a.set_index('Cima')['Err_Rel_Porcentual'].to_dict()
    
    # Lógica de Pozo B
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
    
    tvdss_b0 = results['TVDSS_base'].iloc[0]
    results['Prediccion_Base'] = (results['esp_base'].cumsum().shift(1).fillna(0)) + tvdss_b0
    results['Prop_Sigma_Acum'] = np.sqrt((results['err_prop_capa']**2).cumsum().shift(1).fillna(0))
    results['Prop_Max'] = results['Prediccion_Base'] + results['Prop_Sigma_Acum']
    results['Prop_Min'] = results['Prediccion_Base'] - results['Prop_Sigma_Acum']
    results['Esp_Calc_Neg'] = ((results['esp_base'] - results['ajuste_err']).cumsum().shift(1).fillna(0)) + tvdss_b0
    results['Esp_Calc_Pos'] = ((results['esp_base'] + results['ajuste_err']).cumsum().shift(1).fillna(0)) + tvdss_b0
    
    return df_a, results

# --- 3. INTERFAZ DE USUARIO ---
st.title("⚒️ Herramienta de Pronóstico de Cimas Geológicas")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 1. Pozo de Referencia", 
    "🔮 2. Pozo Pronóstico", 
    "📉 3. Pozo Columna Real", 
    "💾 4. Exportar Datos"
])

# Configuración de las columnas desplegables (Dropdowns)
config_surf = {
    "Surf": st.column_config.SelectboxColumn(
        "Surf",
        options=["Concordante", "Discordante"],
        required=True,
    )
}

config_superficie = {
    "Superficie": st.column_config.SelectboxColumn(
        "Superficie",
        options=["Concordante", "Discordante"],
        required=True,
    )
}

with tab1:
    st.header("Configuración de Pozo de Referencia (A)")
    c1, c2, c3 = st.columns(3)
    dt_val = c1.number_input("Delta T (s):", value=0.002, format="%.3f")
    dv_val = c2.number_input("Delta V (m/s):", value=50.0)
    de_val = c3.number_input("Delta Err:", value=5.0)

    data_a = {
        'Surf': ['Concordante', 'Concordante', 'Concordante', 'Concordante'],
        'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TVDSS (m)': [1848, 2052, 2076, 2116],
        'TWT (s)': [1.305, 1.404, 1.416, 1.434],
        'Vel avg (m/s)': [4135, 4260, 4525, 5157]
    }
    df_a_input = st.data_editor(pd.DataFrame(data_a), num_rows="dynamic", key="editor_a", use_container_width=True, column_config=config_surf)

with tab2:
    st.header("Configuración de Pozo Pronóstico (B)")
    c4, c5 = st.columns(2)
    tvdss_base = c4.number_input("TVDSS Inicial Pozo B (m):", value=1927)
    escenario = c5.selectbox("Escenario de Visualización:", ['todos', 'base', 'somero', 'profundo'])
    show_prop = st.checkbox("Mostrar Incertidumbre Propagada", value=True)

    data_b = {
        'Superficie': ['Concordante', 'Concordante', 'Concordante', 'Concordante'],
        'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TWT Pred (s)': [1.274, 1.314, 1.314, 1.314],
        'Vel avg (m/s)': [4362, 4260, 4525, 5157]
    }
    df_b_input = st.data_editor(pd.DataFrame(data_b), num_rows="dynamic", key="editor_b", use_container_width=True, column_config=config_superficie)

with tab3:
    st.header("Validación con Pozo Real")
    show_real = st.checkbox("Graficar Pozo Real sobre el Pronóstico")
    data_r = {
        'Surf': ['Concordante']*4,
        'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TWT Real (s)': [1.270, 1.400, 1.410, 1.430],
        'TVDSS Real (m)': [1927, 2100, 2150, 2200]
    }
    df_real_input = st.data_editor(pd.DataFrame(data_r), num_rows="dynamic", key="editor_r", use_container_width=True, column_config=config_surf)

with tab4:
    st.header("Exportación de Resultados")
    nombre_ref = st.text_input("Nombre Pozo Referencia:", "Pozo A")
    nombre_pre = st.text_input("Nombre Pozo Pronóstico:", "Pozo B")
    interprete = st.text_input("Nombre del Intérprete:", "")
    
    st.info("Para exportar, primero debe generar el análisis abajo.")

# --- 4. EJECUCIÓN DEL ANÁLISIS ---
st.markdown("---")
if st.button("📊 GENERAR ANÁLISIS COMPLETO", use_container_width=True, type="primary"):
    try:
        # Preparar datos
        df_b_input['TVDSS_base'] = tvdss_base
        
        # Procesar cálculos
        df_a_res, res_b = calcular_predicciones(df_a_input, df_b_input, dt_val, dv_val, de_val)

        # Crear Visualización
        fig, (ax_well, ax_main) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [1, 4], 'wspace': 0.15})
        
        # --- Columna Geológica (Izquierda) ---
        df_well = res_b[res_b['Superficie'] == 'Concordante']
        y_max_plot = res_b['Prop_Max'].max() + 50
        y_min_plot = res_b['Prediccion_Base'].min() - 50

        for i in range(len(df_well)):
            y_top = df_well['Prediccion_Base'].iloc[i]
            y_bottom = df_well['Prediccion_Base'].iloc[i+1] if i+1 < len(df_well) else y_max_plot
            y_mid = (y_top + y_bottom) / 2
            cima_name = df_well['Cima'].iloc[i]
            color_capa = obtener_color(cima_name)
            
            ax_well.axhspan(y_top, y_bottom, facecolor=color_capa, alpha=0.8, edgecolor='black')
            ax_well.text(0.5, y_mid, f"{cima_name}", ha='center', va='center', fontsize=9, fontweight='bold', transform=ax_well.get_yaxis_transform())

        ax_well.set_ylim(y_max_plot, y_min_plot)
        ax_well.set_xticks([])
        ax_well.set_ylabel("Profundidad (TVDSS m)")
        ax_well.set_title("Columna Pronosticada", fontsize=10)

        # --- Gráfico de Predicción (Derecha) ---
        if show_prop and escenario in ['base', 'todos']:
            ax_main.fill_between(res_b['TWT Pred (s)'], res_b['Prop_Min'], res_b['Prop_Max'], color='gray', alpha=0.15, label='Incertidumbre')
        
        if escenario in ['base', 'todos']:
            ax_main.plot(res_b['TWT Pred (s)'], res_b['Prediccion_Base'], 'g-o', label='Pronóstico Base', markersize=8)
        
        if escenario in ['somero', 'todos']:
            ax_main.plot(res_b['TWT Pred (s)'], res_b['Esp_Calc_Neg'], 'r--o', label='Pronóstico Somero')
            
        if escenario in ['profundo', 'todos']:
            ax_main.plot(res_b['TWT Pred (s)'], res_b['Esp_Calc_Pos'], 'b--o', label='Pronóstico Profundo')

        if show_real:
            df_plot_real = df_real_input.dropna(subset=['TWT Real (s)', 'TVDSS Real (m)'])
            ax_main.plot(df_plot_real['TWT Real (s)'], df_plot_real['TVDSS Real (m)'], 'k--x', linewidth=2, label='POZO REAL')

        ax_main.set_ylim(y_max_plot, y_min_plot)
        ax_main.set_xlabel("Tiempo (TWT s)")
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper right')
        ax_main.set_title("Curva de Predicción Geológica", fontsize=12)

        st.pyplot(fig)
        
        # Mostrar tablas de resultados debajo
        st.subheader("📋 Resultados del Pronóstico")
        st.dataframe(res_b[['Cima', 'Prediccion_Base', 'Prop_Min', 'Prop_Max', 'err_prop_capa']], use_container_width=True)

        # Preparar Excel para descarga
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            res_b.to_excel(writer, index=False, sheet_name='Pronostico')
            df_a_res.to_excel(writer, index=False, sheet_name='Referencia')
        
        st.sidebar.download_button(
            label="📥 Descargar Resultados (Excel)",
            data=output.getvalue(),
            file_name=f"Pronostico_{nombre_pre}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
