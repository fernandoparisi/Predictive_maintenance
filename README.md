# 🛠️ *Predictive Maintenance*  

## *Project Objective*  

*The main goal of this project is to **predict failures in industrial machines** in order to avoid unplanned downtime and reduce operational costs.*  
*By analyzing sensor data and applying **machine learning** techniques, patterns that anticipate potential failures are identified, enabling **preventive, proactive, and efficient** maintenance.*  

*The solution was deployed through an interactive application built with **Streamlit**, allowing easy visualization of results and usage of the predictive models.*  

---

## *Files*  

- **predictive_maintenance.ipynb**: *Notebook containing the complete project development.*🔗 [*View notebook*](https://github.com/fernandoparisi/Predictive_maintenance/blob/main/predictive_maintenance.ipynb)
- **predictive_maintenance.csv**: *Dataset used, obtained from Kaggle.* 🔗 [*View dataset on Kaggle*](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)  
- **app.py**: *Base code of the Streamlit application.* 🔗 [*Access the app*](https://predictivemaintenance-parisi.streamlit.app/)  

---

## *Project Summary*  

- **Exploratory Data Analysis (EDA):**  
  *A detailed analysis of the dataset was performed to understand the variables, their distribution, and relationships.*
![1757450880254](https://github.com/user-attachments/assets/c0c6b6f8-7115-447a-85f2-2b0862a0a27b)
![1757450880115](https://github.com/user-attachments/assets/48b72efa-9623-4a04-90dd-423055feaeec)

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
    
![Machine Learning Prediction](https://github.com/user-attachments/assets/b57758ad-3efc-43ee-b9ae-aa302006b989)


