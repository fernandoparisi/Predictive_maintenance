# 🛠️ Mantenimiento Predictivo  

## 🎯 Objetivo del Proyecto  

El objetivo principal de este proyecto es **predecir fallas en máquinas industriales** con el fin de evitar paradas no planificadas y reducir los costos operativos.  
Mediante el análisis de datos provenientes de sensores y la aplicación de técnicas de **machine learning**, se identifican patrones que anticipan una posible falla, lo que permite implementar un mantenimiento **preventivo, proactivo y eficiente**.  

La solución fue desplegada a través de una aplicación interactiva desarrollada en **Streamlit**, que facilita la visualización de resultados y la utilización de los modelos predictivos.  

---

## 📂 Archivos  

- **predictive_maintenance.ipynb**: Notebook con el desarrollo completo del proyecto.  
- **predictive_maintenance.csv**: Dataset utilizado, obtenido de Kaggle.  
  🔗 [Ver dataset en Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)  
- **app.py**: Código base de la aplicación en Streamlit.  
  🔗 [Acceso a la aplicación](https://predictivemaintenance-parisi.streamlit.app/)  

---

## 📊 Resumen del Proyecto  

- **Análisis Exploratorio de Datos (EDA):**  
  Se realizó un estudio detallado del dataset para comprender las variables, su distribución y las relaciones entre ellas.  

- **Ingeniería de Características:**  
  Se generaron nuevas variables con el objetivo de mejorar la capacidad predictiva de los modelos.  

- **Modelado de Machine Learning:**  
  Se entrenaron y evaluaron cuatro modelos de **clasificación binaria:**  
  1. Regresión Logística  
  2. Random Forest  
  3. Gradient Boosting  
  4. XGBoost  

  También se exploró un modelo de **clasificación multiclase**, pero fue descartado debido al **alto desbalance entre clases**.  

- **Resultados y Evaluación:**  
  El modelo **XGBoost** obtuvo el mejor desempeño, alcanzando un **F1-Score de 0.72** en la clase *“Falla”*.  

- **Aplicación Web Interactiva:**  
  Se construyó una app en **Streamlit** para consultar predicciones en tiempo real y visualizar el rendimiento de los modelos.  

---


