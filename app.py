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
# Usamos @st.cache_resource para que esta función solo se ejecute una vez y almacene los modelos.
@st.cache_resource
def train_models(X_train, y_train):
    """
    Entrena y devuelve un diccionario de modelos de aprendizaje automático binarios.
    Esta función se ejecuta solo una vez.
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

    # Entrenar todos los modelos
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)

    return pipelines

# Llamar a la función de entrenamiento. La primera vez tomará tiempo, luego será rápido.
with st.spinner('Entrenando modelos (esto solo ocurre la primera vez)...'):
    pipelines = train_models(X_train, y_train)

st.success('¡Entrenamiento de modelos completado!')


# --- 3. DICCIONARIOS DE IDIOMAS ---
text_strings = {
    'es': {
        'title': 'Predicción de Fallas en Máquinas 🛠️',
        'subtitle': 'Seleccione un modelo y ajuste los parámetros de la máquina.',
        'select_model': 'Seleccionar Modelo',
        'section_params': 'Parámetros de la Máquina',
        'type': 'Tipo de Producto',
        'air_temp': 'Temperatura del Aire [K]',
        'process_temp': 'Temperatura del Proceso [K]',
        'rotational_speed': 'Velocidad de Rotación [rpm]',
        'torque': 'Torque [Nm]',
        'tool_wear': 'Desgaste de Herramienta [min]',
        'predict_button': 'Predecir Falla',
        'prediction_title': 'Resultado de la Predicción',
        'failure_detected': '¡PREDICCIÓN: FALLA DETECTADA!',
        'no_failure': 'PREDICCIÓN: No hay falla.',
        'loading': 'Entrenando modelos (esto solo ocurre la primera vez)...',
        'success_loading': '¡Entrenamiento de modelos completado!'
    },
    'en': {
        'title': 'Machine Failure Prediction 🛠️',
        'subtitle': 'Select a model and adjust the machine parameters.',
        'select_model': 'Select Model',
        'section_params': 'Machine Parameters',
        'type': 'Product Type',
        'air_temp': 'Air Temperature [K]',
        'process_temp': 'Process Temperature [K]',
        'rotational_speed': 'Rotational Speed [rpm]',
        'torque': 'Torque [Nm]',
        'tool_wear': 'Tool Wear [min]',
        'predict_button': 'Predict Failure',
        'prediction_title': 'Prediction Result',
        'failure_detected': 'PREDICTION: FAILURE DETECTED!',
        'no_failure': 'PREDICTION: No failure.',
        'loading': 'Training models (this only happens the first time)...',
        'success_loading': 'Model training completed!'
    }
}

# --- 4. INTERFAZ DE STREAMLIT ---
# Selector de idioma
lang = st.selectbox(
    "Select Language / Seleccionar Idioma",
    ('es', 'en'),
    format_func=lambda x: 'Español' if x == 'es' else 'English'
)

# Título y subtítulo
st.markdown(f"## {text_strings[lang]['title']}") # CAMBIO AQUÍ: USANDO MARKDOWN PARA UN TAMAÑO MÁS GRANDE QUE ANTES
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>
    <h3 style='text-align: center;'>{text_strings[lang]['subtitle']}</h3>
</div>
""", unsafe_allow_html=True)

# Selección del modelo
model_option = st.selectbox(
    text_strings[lang]['select_model'],
    ('XGBoost', 'LightGBM', 'Bosque Aleatorio', 'Regresión Logística')
)

st.markdown("---")

# Entradas de usuario con sliders
st.subheader(text_strings[lang]['section_params'])

col1, col2, col3 = st.columns(3)

with col1:
    tipo = st.selectbox(text_strings[lang]['type'], ('L', 'M', 'H'))
    temp_aire = st.slider(text_strings[lang]['air_temp'], min_value=290.0, max_value=310.0, value=298.6, step=0.1)

with col2:
    temp_proceso = st.slider(text_strings[lang]['process_temp'], min_value=300.0, max_value=320.0, value=308.6, step=0.1)
    vel_rotacion = st.slider(text_strings[lang]['rotational_speed'], min_value=800, max_value=3000, value=1000)

with col3:
    torque = st.slider(text_strings[lang]['torque'], min_value=0.0, max_value=80.0, value=40.0, step=0.1)
    desgaste_herramienta = st.slider(text_strings[lang]['tool_wear'], min_value=0, max_value=300, value=0)


# Botón de predicción
if st.button(text_strings[lang]['predict_button'], key='predict_button'):
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
    st.subheader(text_strings[lang]['prediction_title'])
    if prediccion == 1:
        st.error(text_strings[lang]['failure_detected'])
    else:
        st.success(text_strings[lang]['no_failure'])
