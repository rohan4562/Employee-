# HR Attrition Prediction Using Machine Learning

A complete end-to-end machine learning project that predicts employee attrition using HR analytics data. It includes data preprocessing, model building, evaluation, and deployment with a user-friendly interface using **Streamlit**.

---

## 📌 Problem Statement

Employee attrition, or voluntary resignation, is a significant issue for organizations. Understanding why employees leave enables HR teams to implement strategic interventions and improve employee retention.

**Objective:** Build a predictive model to determine whether an employee is likely to leave the company, based on various job and personal attributes.

---

## 🔍 Key Features Used

* Age
* Business Travel Frequency
* Environment Satisfaction
* Job Satisfaction
* Monthly Income
* Stock Option Level
* Total Working Years
* Years at Company
* OverTime
* Years with Current Manager

*(Final feature list may vary after preprocessing and feature selection)*

---

## ⚙️ Technologies and Libraries

| Technology          | Role                             |
| ------------------- | -------------------------------- |
| Python              | Programming language             |
| Pandas, NumPy       | Data manipulation                |
| Matplotlib, Seaborn | Data visualization               |
| Scikit-learn        | Model building and evaluation    |
| imbalanced-learn    | Handling class imbalance (SMOTE) |
| Joblib              | Model saving/loading             |
| Streamlit           | Interactive web app deployment   |

---

## 📊 Project Workflow

### 1. Data Preprocessing

* Cleaned and encoded categorical variables
* Handled missing values and outliers

### 2. Exploratory Data Analysis (EDA)

* Visualized key patterns in attrition, income, overtime, etc.

### 3. Class Imbalance Handling

* Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset

### 4. Model Training

* Algorithm: **Random Forest Classifier**
* Evaluation Metrics: Accuracy, Confusion Matrix, ROC-AUC Score

### 5. Model Evaluation

* **Accuracy**: \~87%
* **ROC-AUC Score**: \~0.72 (Post SMOTE)

### 6. Streamlit Deployment

* Built an interactive web app to take employee details and predict attrition likelihood in real-time

---

---

## 🔧 How to Run Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/hr-attrition-prediction.git
cd hr-attrition-prediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit App

```bash
streamlit run hr_attrition_app.py
```

---

## 📁 Project Structure

```
├── data/
│   └── HR_Employee_Attrition.csv
├── hr_attrition_app.py       # Streamlit app code
├── model/
│   └── hr_model.pkl          # Trained model file
├── notebook/
│   └── HR_Attrition_EDA_Model.ipynb
├── requirements.txt
└── README.md
```

---

---

## 🤔 Future Enhancements

* Integrate SHAP or LIME for model explainability
* Experiment with advanced algorithms (e.g., XGBoost, LightGBM)
* Add authentication and logging for enterprise usage

---

## 👤 Author

Rohan chomal
---

## 📄 License

This project is licensed under the MIT License.
