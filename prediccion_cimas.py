import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE COLORES (Tu lógica original) ---
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
        color_norm = tuple(c / 255 for c in rgb)
        for p in palabras:
            if p.lower() in nombre_clean: return color_norm
    return (0.8, 0.8, 0.8)

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Geología Predictiva", layout="wide")
st.title("⚒️ Pronóstico de Cimas Geológicas")

with st.sidebar:
    st.header("Configuración de Error")
    w_dt = st.number_input("Delta T (s)", value=0.002, format="%.3f")
    w_dv = st.number_input("Delta V (m/s)", value=50.0)
    w_de = st.number_input("Delta Err", value=5.0)
    tvdss_init = st.number_input("TVDSS Inicial (m)", value=1927)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Pozo de Referencia")
    df_a = st.data_editor(pd.DataFrame({
        'Surf': ['Concordante']*4,
        'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TVDSS (m)': [1848, 2052, 2076, 2116],
        'TWT (s)': [1.305, 1.404, 1.416, 1.434],
        'Vavg (m/s)': [4135, 4260, 4525, 5157]
    }), num_rows="dynamic", key="tabla_a")

with col2:
    st.subheader("2. Pozo Pronóstico")
    df_b = st.data_editor(pd.DataFrame({
        'Superficie': ['Concordante']*4,
        'Cima': ['KS Mendez', 'KS San Felipe', 'Ks Agua Nueva', 'KM Tamabra'],
        'TWT Pred (s)': [1.274, 1.314, 1.314, 1.314],
        'Vavg (m/s)': [4362, 4260, 4525, 5157]
    }), num_rows="dynamic", key="tabla_b")

if st.button("🚀 CALCULAR Y GRAFICAR", use_container_width=True):
    # (Aquí iría tu función calcular_predicciones conectada a df_a y df_b)
    st.success("Cálculos realizados con éxito")
    
    # Ejemplo rápido de gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_b['TWT Pred (s)'], df_b['Vavg (m/s)'], 'g-o')
    ax.invert_yaxis()
    st.pyplot(fig)