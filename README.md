# 🛠️Mantenimiento Predictivo
# Objetivo del proyecto 

Este proyecto tiene como objetivo principal predecir la falla de máquinas industriales para evitar paradas indeseadas y reducir los costos de operación. A través del análisis de datos de sensores y el uso de técnicas de **machine learning**, se busca identificar los patrones que indican una falla inminente, permitiendo un mantenimiento preventivo, proactivo y eficiente.
La solución se implementó con una aplicación interactiva usando **Streamlit**, facilitando la visualización y el uso de los modelos predictivos.
# Resumen del proyecto
**Análisis Exploratorio de Datos (EDA):** Se realizó un análisis detallado del dataset de mantenimiento predictivo para entender las variables y la distribución de los datos.

**Ingeniería de Características:** Se crearon nuevas variables para mejorar la capacidad predictiva de los modelos.

**Modelado de Machine Learning:** Se evaluaron cuatro modelos de **clasificación binaria:**
- Regresión Logística
- Random Forest
- Gradient Boosting
- XGBoost

**Resultados y Evaluación:** El modelo de XGBoost se destacó como el de mejor rendimiento, logrando un **F1-Score de 0.72** en la clase "Falla".

**Aplicación Web Interactiva:** Una app construida con **Streamlit** para visualizar las predicciones y el desempeño del modelo.
