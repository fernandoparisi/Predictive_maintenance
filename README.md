# 🛠️ Mantenimiento Predictivo  

## Objetivo del Proyecto  

El objetivo principal de este proyecto es **predecir fallas en máquinas industriales** con el fin de evitar paradas no planificadas y reducir los costos operativos.  
Mediante el análisis de datos provenientes de sensores y la aplicación de técnicas de **machine learning**, se identifican patrones que anticipan una posible falla, lo que permite implementar un mantenimiento **preventivo, proactivo y eficiente**.  

La solución fue desplegada a través de una aplicación interactiva desarrollada en **Streamlit**, que facilita la visualización de resultados y la utilización de los modelos predictivos.  

---

## Archivos  

- **predictive_maintenance.ipynb**: Notebook con el desarrollo completo del proyecto. 🔗 [Ver notebook](https://github.com/fernandoparisi/Predictive_maintenance/blob/main/predictive_maintenance.ipynb)
- **predictive_maintenance.csv**: Dataset utilizado, obtenido de Kaggle. 🔗 [Ver dataset en Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)  
- **app.py**: Código base de la aplicación en Streamlit. 🔗 [Acceso a la aplicación](https://predictivemaintenance-parisi.streamlit.app/)  

---

## Resumen del Proyecto  

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
# 🛠️ *Predictive Maintenance*  

## *Project Objective*  

*The main goal of this project is to **predict failures in industrial machines** in order to avoid unplanned downtime and reduce operational costs.*  
*By analyzing sensor data and applying **machine learning** techniques, patterns that anticipate potential failures are identified, enabling **preventive, proactive, and efficient** maintenance.*  

*The solution was deployed through an interactive application built with **Streamlit**, allowing easy visualization of results and usage of the predictive models.*  

---

## *Files*  

- **predictive_maintenance.ipynb**: *Notebook containing the complete project development.*
  🔗 [*View notebook*](https://github.com/fernandoparisi/Predictive_maintenance/blob/main/predictive_maintenance.ipynb)
- **predictive_maintenance.csv**: *Dataset used, obtained from Kaggle.*  
  🔗 [*View dataset on Kaggle*](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)  
- **app.py**: *Base code of the Streamlit application.*  
  🔗 [*Access the app*](https://predictivemaintenance-parisi.streamlit.app/)  

---

## *Project Summary*  

- **Exploratory Data Analysis (EDA):**  
  *A detailed analysis of the dataset was performed to understand the variables, their distribution, and relationships.*  

- **Feature Engineering:**  
  *New variables were created to improve the predictive power of the models.*  

- **Machine Learning Modeling:**  
  *Four **binary classification** models were trained and evaluated:*  
  1. *Logistic Regression*  
  2. *Random Forest*  
  3. *Gradient Boosting*  
  4. *XGBoost*  

  *A **multiclass classification** approach was also explored but discarded due to **severe class imbalance**.*  

- **Results and Evaluation:**  
  *The **XGBoost** model achieved the best performance, reaching an **F1-Score of 0.72** for the *“Failure”* class.*  

- **Interactive Web Application:**  
  *An app was built with **Streamlit** to query real-time predictions and visualize model performance.*  


