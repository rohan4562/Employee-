import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model, feature_names = joblib.load("hr_model.pkl")

st.title("HR Attrition Prediction App")

# Input form for user to fill
st.header("Enter Employee Details")

age = st.slider("Age", 18, 60, 30)
business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education = st.selectbox("Education (1: Below College, 5: Doctor)", [1, 2, 3, 4, 5])
education_field = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
monthly_income = st.number_input("Monthly Income", min_value=1000, step=100)
stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
total_working_years = st.slider("Total Working Years", 0, 40, 5)
years_at_company = st.slider("Years At Company", 0, 30, 2)
years_with_manager = st.slider("Years With Current Manager", 0, 30, 2)
over_time = st.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
env_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])

# Create input DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "BusinessTravel": business_travel,
    "Department": department,
    "Education": education,
    "EducationField": education_field,
    "Gender": gender,
    "JobRole": job_role,
    "MaritalStatus": marital_status,
    "MonthlyIncome": monthly_income,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "YearsWithCurrManager": years_with_manager,
    "OverTime": over_time,
    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": env_satisfaction
}])

# One-hot encoding (match what was done during model training)
input_data = pd.get_dummies(input_data)

# Align input columns with training features
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. (Attrition Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Attrition Probability: {prediction_proba:.2f})")
