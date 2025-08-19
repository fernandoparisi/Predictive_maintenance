# app.py

import streamlit as st
import pandas as pd
import joblib

# --- 1. Cargar el modelo y las columnas del disco ---
# Asegúrate de que estos archivos estén en la misma carpeta que app.py
try:
    modelo = joblib.load('modelo_random_forest.joblib')
    columnas = joblib.load('columnas_entrenamiento.joblib')
except FileNotFoundError:
    st.error("Error: Archivos del modelo no encontrados. Asegúrate de que 'modelo_random_forest.joblib' y 'columnas_entrenamiento.joblib' estén en la misma carpeta.")

# --- 2. Función de predicción con el modelo cargado ---
def predecir_falla(modelo_cargado, columnas_cargadas, tipo, temp_proceso, vel_rotacion, desgaste_herramienta):
    """
    Función para predecir si una máquina fallará o no.
    """
    datos_entrada = {
        'Type': [tipo],
        'Process temperature [K]': [temp_proceso],
        'Rotational speed [rpm]': [vel_rotacion],
        'Tool wear [min]': [desgaste_herramienta]
    }
    df_entrada = pd.DataFrame(datos_entrada)
    df_entrada_encoded = pd.get_dummies(df_entrada)
    
    # Alineamos el DataFrame de entrada con las columnas de entrenamiento
    df_entrada_alineado = df_entrada_encoded.reindex(columns=columnas_cargadas, fill_value=0)
    
    prediccion = modelo_cargado.predict(df_entrada_alineado)
    return prediccion[0]

# --- 3. Interfaz de Usuario con Streamlit ---
st.title('Aplicación de Mantenimiento Predictivo ⚙️')
st.markdown('### Ingresa los parámetros de la máquina')

# Campos de entrada interactivos
tipo_producto = st.selectbox('Tipo de Producto', ['L', 'M', 'H'])
temp_proceso = st.slider('Temperatura del Proceso [K]', min_value=290.0, max_value=320.0, value=308.6)
vel_rotacion = st.slider('Velocidad de Rotación [rpm]', min_value=1000, max_value=3000, value=1200)
desgaste_herramienta = st.slider('Desgaste de la Herramienta [min]', min_value=0, max_value=255, value=100)

# Botón para hacer la predicción
if st.button('Predecir'):
    if 'modelo' in locals() and 'columnas' in locals():
        # Llamar a la función con los valores de la interfaz
        resultado = predecir_falla(modelo, columnas, tipo_producto, temp_proceso, vel_rotacion, desgaste_herramienta)
        
        st.markdown('---')
        st.subheader('Resultado de la Predicción')
        
        if resultado == 1:
            st.error('🚨 ¡Alerta! El modelo predice una POSIBLE FALLA.')
        else:
            st.success('✅ El modelo predice que NO HABRÁ FALLA.')