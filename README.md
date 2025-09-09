# 🛠️Mantenimiento Predictivo
# Objetivo del proyecto 

Este proyecto tiene como objetivo principal predecir la falla de máquinas industriales para evitar paradas indeseadas y reducir los costos de operación. A través del análisis de datos de sensores y el uso de técnicas de **machine learning**, se busca identificar los patrones que indican una falla inminente, permitiendo un mantenimiento preventivo, proactivo y eficiente.
La solución se implementó con una aplicación interactiva usando **Streamlit**, facilitando la visualización y el uso de los modelos predictivos.

# Archivos 
- **predictive_maintenance.ipynb**: Notebook donde se desarrolló el proyecto. 
- **predictive_maintenance.csv**: Dataset obtenido de Kaggle. Link: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
- **app.py**: Archivo base de la app creada en Streamlit. Link: https://predictivemaintenance-parisi.streamlit.app/ 

# Resumen del proyecto
- **Análisis Exploratorio de Datos (EDA):** Se realizó un análisis detallado del dataset de mantenimiento predictivo para entender las variables y la distribución de los datos.

- **Ingeniería de Características:** Se crearon nuevas variables para mejorar la capacidad predictiva de los modelos.

- **Modelado de Machine Learning:** Se evaluaron cuatro modelos de **clasificación binaria:**
  1. Regresión Logística
  2. Random Forest
  3. Gradient Boosting
  4. XGBoost

Se analizó el modelo de **clasificación multiclase**, pero se descartó el análisis del mismo por **desbalances de clases**

- **Resultados y Evaluación:** El modelo de XGBoost se destacó como el de mejor rendimiento, logrando un **F1-Score de 0.72** en la clase "Falla".

- **Aplicación Web Interactiva:** Una app construida con **Streamlit** para visualizar las predicciones y el desempeño del modelo.


🛠️ Mantenimiento Predictivo
Objetivo del Proyecto
Este proyecto tiene como objetivo principal predecir la falla de máquinas industriales para evitar paradas no planificadas y reducir significativamente los costos de operación. A través del análisis de datos de sensores y el uso de machine learning, se busca identificar los patrones que indican una falla inminente, permitiendo un mantenimiento proactivo y eficiente.

La solución se implementó como una aplicación web interactiva usando Streamlit, lo que facilita la visualización y el uso de los modelos predictivos.

Contenido del Repositorio
predictive_maintenance.ipynb: El notebook principal con todo el análisis exploratorio de datos (EDA), la ingeniería de características y el desarrollo de los modelos de machine learning.

predictive_maintenance.csv: El conjunto de datos de mantenimiento predictivo, obtenido de Kaggle. Enlace al Dataset

app.py: El archivo base de la aplicación web creada en Streamlit. Puedes ver la versión desplegada en el siguiente Enlace a la App
