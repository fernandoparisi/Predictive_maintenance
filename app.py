import streamlit as st
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE DATOS ---
try:
    df = pd.read_csv('predictive_maintenance.csv')
    df_bin = df.drop(['Failure Type'], axis=1)
    X = df_bin.drop(['Target', 'Product ID'], axis=1)
    y = df_bin['Target']

    # Separar los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except FileNotFoundError:
    st.error("Error: Archivo 'predictive_maintenance.csv' no encontrado. Asegúrate de que está en la misma carpeta que tu script.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()


# --- 2. DEFINICIÓN Y ENTRENAMIENTO DE LOS MODELOS (AHORA EN CACHÉ) ---
# Usamos @st.cache_data para que esta función solo se ejecute una vez.
# Esto hace que el entrenamiento sea instantáneo en las ejecuciones posteriores.
@st.cache_data
def train_models(X_train, y_train):
    """
    Entrena y devuelve un diccionario de modelos de aprendizaje automático.
    Esta función se ejecuta solo una vez gracias a st.cache_data.
    """
    # Preprocesador para la columna 'Type'
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Type'])
        ],
        remainder='passthrough'
    )

    # Diccionario de pipelines con los modelos
    pipelines = {
        'Regresión Logística': Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        'Bosque Aleatorio': Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ]),
        'LightGBM': Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LGBMClassifier(random_state=42))
        ])
    }

    # Entrenar todos los modelos y devolver los pipelines
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)

    return pipelines

# Llamar a la función de entrenamiento. La primera vez tomará tiempo, luego será rápido.
with st.spinner('Entrenando modelos (esto solo ocurre la primera vez)...'):
    pipelines = train_models(X_train, y_train)

st.success('¡Entrenamiento de modelos completado!')


# --- 3. INTERFAZ DE STREAMLIT ---
# Usamos columnas para un control preciso del título y el emoji
col_empty1, col_title, col_icon, col_empty2 = st.columns([2, 5, 1, 2])
with col_title:
    st.markdown("<h2 style='text-align: right;'>Predicción de Fallas en Máquinas</h2>", unsafe_allow_html=True)
with col_icon:
    st.markdown("## 🛠️")

st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>
    <h3 style='text-align: center;'>Seleccione un modelo y ajuste los parámetros de la máquina.</h3>
</div>
""", unsafe_allow_html=True)

# Selección del modelo
model_option = st.selectbox(
    'Seleccionar Modelo',
    ('XGBoost', 'LightGBM', 'Bosque Aleatorio', 'Regresión Logística')
)

st.markdown("---")

# Entradas de usuario con sliders
st.subheader("Parámetros de la Máquina")

col1, col2, col3 = st.columns(3)

with col1:
    tipo = st.selectbox('Tipo de Producto', ('L', 'M', 'H'))
    temp_aire = st.slider('Temperatura del Aire [K]', min_value=290.0, max_value=310.0, value=298.6, step=0.1)

with col2:
    temp_proceso = st.slider('Temperatura del Proceso [K]', min_value=300.0, max_value=320.0, value=308.6, step=0.1)
    vel_rotacion = st.slider('Velocidad de Rotación [rpm]', min_value=800, max_value=3000, value=1000)

with col3:
    torque = st.slider('Torque [Nm]', min_value=0.0, max_value=80.0, value=40.0, step=0.1)
    desgaste_herramienta = st.slider('Desgaste de Herramienta [min]', min_value=0, max_value=300, value=0)


# Botón de predicción
if st.button('Predecir Falla', key='predict_button'):
    # Crear un DataFrame con los datos de entrada
    datos_entrada = pd.DataFrame({
        'UDI': [1234],
        'Type': [tipo],
        'Air temperature [K]': [temp_aire],
        'Process temperature [K]': [temp_proceso],
        'Rotational speed [rpm]': [vel_rotacion],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [desgaste_herramienta]
    })

    # Obtener la predicción del modelo seleccionado
    modelo_seleccionado = pipelines[model_option]
    prediccion = modelo_seleccionado.predict(datos_entrada)[0]

    # Mostrar el resultado
    st.markdown("---")
    if prediccion == 1:
        st.error('¡PREDICCIÓN: FALLA DETECTADA!')
    else:
        st.success('PREDICCIÓN: No hay falla.')