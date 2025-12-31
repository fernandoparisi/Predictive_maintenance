## 🛠️ Predictive Maintenance Analytics for Industrial Equipment

This project delivers an **end-to-end predictive maintenance solution** designed to reduce unplanned downtime and optimize maintenance decisions for industrial equipment.

The solution is deployed through an **interactive web application built with Streamlit**, allowing users to simulate operating conditions and obtain real-time failure predictions based on machine learning models.

![Machine Learning Prediction](https://github.com/user-attachments/assets/2242b72a-1522-4a0d-ba4d-818d91b25d1a)

---

## 🚀 Interactive Web Application

The Streamlit application allows users to:
- Select a predictive model  
- Adjust machine operating parameters  
- Obtain real-time failure predictions  
- Visualize model outputs in an intuitive way  

This interface bridges the gap between advanced analytics and operational decision-making.

---

## 🎯 Business Objective

The main goal of this project is to **predict machine failures before they occur**, enabling:
- Reduced unplanned downtime  
- Lower maintenance and operational costs  
- More proactive and data-driven maintenance planning  

By analyzing sensor data and applying machine learning techniques, the project identifies patterns that anticipate potential failures under specific operating conditions.

---

## 📊 Exploratory Data Analysis (EDA)

### Correlation Analysis
![1757450880115](https://github.com/user-attachments/assets/927eae39-57e0-4715-aa1f-69c085871976)

A correlation analysis was conducted to identify relationships between operational variables.  
Strong correlations were observed between:
- Process temperature and air temperature  
- Torque and rotational speed  

These relationships highlight key operational dependencies and potential multicollinearity, informing feature selection and model choice.

---

### Feature Relationships and Failure Patterns
![1757450880254](https://github.com/user-attachments/assets/704c7268-2cc6-417a-9538-d188f072b402)

Scatter plots colored by failure occurrence were used to explore how different operating conditions relate to machine failures.  
Although failures are relatively rare, certain regions of the feature space show higher concentrations of failure events, supporting the use of non-linear models.

---

## 🧠 Feature Engineering

New features were created to enhance the predictive power of the models, capturing interactions between operational variables and improving failure detection capability.

---

## 🤖 Machine Learning Modeling

Four binary classification models were trained and evaluated:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

A multiclass classification approach was also explored but ultimately discarded due to severe class imbalance and limited practical value for operational decision-making.

---

## 📈 Results and Evaluation

The **XGBoost model** achieved the best performance, reaching an **F1-Score of 0.72 for the “Failure” class**.

The evaluation prioritized performance on rare but critical failure events, making the model suitable for predictive maintenance scenarios where missing a failure is more costly than false positives.

---

## 📁 Project Files

- `predictive_maintenance.ipynb` – Full project development notebook  
- `predictive_maintenance.csv` – Dataset obtained from Kaggle  
- `app.py` – Streamlit application source code  

---

## 🛠️ Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  
- Streamlit  



