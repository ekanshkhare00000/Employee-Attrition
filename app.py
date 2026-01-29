import streamlit as st
import numpy as np
import joblib

# ---------------- Load Model & Scaler ----------------
model = joblib.load("sale.pkl")      # your trained attrition model
scaler = joblib.load("scaler.pkl")   # StandardScaler

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("👨‍💼 Employee Attrition Prediction App")

st.write("Fill employee details to predict attrition")

# ---------------- Categorical Mappings ----------------
BusinessTravel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
Department_map = {"Sales": 0, "Research & Development": 1, "Human Resources": 2}
EducationField_map = {"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Human Resources": 4, "Other": 5}
Gender_map = {"Female": 0, "Male": 1}
JobRole_map = {
    "Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2,
    "Manufacturing Director": 3, "Healthcare Representative": 4,
    "Manager": 5, "Sales Representative": 6, "Research Director": 7, "Human Resources": 8
}
MaritalStatus_map = {"Single": 0, "Married": 1, "Divorced": 2}
Over18_map = {"Y": 1}
OverTime_map = {"No": 0, "Yes": 1}

# ---------------- Inputs ----------------
Age = st.number_input("Age", min_value=18, max_value=65)
BusinessTravel = st.selectbox("Business Travel", list(BusinessTravel_map.keys()))
DailyRate = st.number_input("Daily Rate")
Department = st.selectbox("Department", list(Department_map.keys()))
DistanceFromHome = st.number_input("Distance From Home")
Education = st.selectbox("Education (1-5)", [1,2,3,4,5])
EducationField = st.selectbox("Education Field", list(EducationField_map.keys()))
EmployeeCount = st.number_input("Employee Count", value=1)
EmployeeNumber = st.number_input("Employee Number")
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1-4)", [1,2,3,4])
Gender = st.selectbox("Gender", list(Gender_map.keys()))
HourlyRate = st.number_input("Hourly Rate")
JobInvolvement = st.selectbox("Job Involvement (1-4)", [1,2,3,4])
JobLevel = st.selectbox("Job Level (1-5)", [1,2,3,4,5])
JobRole = st.selectbox("Job Role", list(JobRole_map.keys()))
JobSatisfaction = st.selectbox("Job Satisfaction (1-4)", [1,2,3,4])
MaritalStatus = st.selectbox("Marital Status", list(MaritalStatus_map.keys()))
MonthlyIncome = st.number_input("Monthly Income")
MonthlyRate = st.number_input("Monthly Rate")
NumCompaniesWorked = st.number_input("Num Companies Worked")
Over18 = st.selectbox("Over 18", list(Over18_map.keys()))
OverTime = st.selectbox("Over Time", list(OverTime_map.keys()))
PercentSalaryHike = st.number_input("Percent Salary Hike")
PerformanceRating = st.selectbox("Performance Rating (1-4)", [1,2,3,4])
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1-4)", [1,2,3,4])
StandardHours = st.number_input("Standard Hours", value=80)
StockOptionLevel = st.selectbox("Stock Option Level (0-3)", [0,1,2,3])
TotalWorkingYears = st.number_input("Total Working Years")
TrainingTimesLastYear = st.number_input("Training Times Last Year")
WorkLifeBalance = st.selectbox("Work Life Balance (1-4)", [1,2,3,4])
YearsAtCompany = st.number_input("Years At Company")
YearsInCurrentRole = st.number_input("Years In Current Role")
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion")
YearsWithCurrManager = st.number_input("Years With Current Manager")

# ---------------- Predict ----------------
if st.button("Predict Attrition"):
    input_data = np.array([[  
        Age,
        BusinessTravel_map[BusinessTravel],
        DailyRate,
        Department_map[Department],
        DistanceFromHome,
        Education,
        EducationField_map[EducationField],
        EmployeeCount,
        EmployeeNumber,
        EnvironmentSatisfaction,
        Gender_map[Gender],
        HourlyRate,
        JobInvolvement,
        JobLevel,
        JobRole_map[JobRole],
        JobSatisfaction,
        MaritalStatus_map[MaritalStatus],
        MonthlyIncome,
        MonthlyRate,
        NumCompaniesWorked,
        Over18_map[Over18],
        OverTime_map[OverTime],
        PercentSalaryHike,
        PerformanceRating,
        RelationshipSatisfaction,
        StandardHours,
        StockOptionLevel,
        TotalWorkingYears,
        TrainingTimesLastYear,
        WorkLifeBalance,
        YearsAtCompany,
        YearsInCurrentRole,
        YearsSinceLastPromotion,
        YearsWithCurrManager
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    result = "Yes (Employee will leave)" if prediction == 1 else "No (Employee will stay)"

    st.success(f"Prediction: {result}")
    st.info(f"Probability → No: {prob[0]*100:.2f}% | Yes: {prob[1]*100:.2f}%")
