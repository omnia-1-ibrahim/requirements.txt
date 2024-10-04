import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# إعدادات الصفحة
st.set_page_config(page_title="Telecom Churn Analysis", layout="wide")

# تخصيص الألوان
primary_color = "#1f77b4"
background_color = "#f0f2f5"

# تنسيق الخلفية
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: {background_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {primary_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# تحميل مجموعة البيانات
data_url = "https://raw.githubusercontent.com/omnia-1-ibrahim/requirements.txt/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(data_url)

# الصفحة الرئيسية
st.title("Telecom Churn Analysis and Prediction")
st.image("dataset-cover.png", use_column_width=True)

# عرض البيانات الخام
st.subheader('Raw Data')
st.write(data.head())

# تحليل البيانات باستخدام الرسوم البيانية
st.subheader('Churn Analysis')
churn_count = data['Churn'].value_counts()
st.bar_chart(churn_count)

# إنشاء نموذج الإدخال
st.subheader("Churn Prediction Input Form")

# Inputs for Machine Learning model
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependent = st.selectbox("Dependent", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 100)
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, step=0.1)
total_charges = st.slider("Total Charges", 0.0, 8000.0, step=0.1)

# Collecting input data
input_data = {
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependent": dependent,
    "Tenure": tenure,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Backup": online_backup,
    "Online Security": online_security,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract Type": contract_type,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
}

# Label Encoding
label_encoder = LabelEncoder()
for feature in ["Senior Citizen", "Partner", "Dependent", "Multiple Lines", "Internet Service", 
                "Online Backup", "Online Security", "Device Protection", "Tech Support", 
                "Streaming TV", "Streaming Movies", "Contract Type", "Paperless Billing", "Payment Method"]:
    input_data[feature] = label_encoder.fit_transform([input_data[feature]])[0]

# Display the encoded inputs
st.write("Encoded Inputs for ML model:", input_data)

# Load or train your ML model (You need to train the model first in your workflow)
# For simplicity, here we are retraining the model
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier()
}

for model in models.values():
    model.fit(X_train, y_train)

# Predict button
if st.button("Predict Churn"):
    # Create a DataFrame for input data to predict
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    predictions = models["Random Forest"].predict(input_df)  # Example using Random Forest

    # Display prediction result
    if predictions[0] == 1:
        st.success("The customer is predicted to churn.")
    else:
        st.success("The customer is predicted not to churn.")

# عرض معلومات الفريق
st.subheader("Meet Our Team")
team_members = [
    {
        "name": "Omnia Ibrahim Sayed",
        "linkedin": "https://www.linkedin.com/in/omnia-ibrahim-8168b022b"
    },
    {
        "name": "Yossef Mohamed Mohamed",
        "linkedin": "https://www.linkedin.com/in/yousef-mohamed-8a4132221/"
    },
    {
        "name": "Abdelrahman Sherif Kamel",
        "linkedin": "http://linkedin.com/in/abdelrahman-sherif-203b66198"
    }
]

for member in team_members:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="margin-right: 10px; font-size: 18px;">{member['name']}</span> 
            <a href="{member['linkedin']}" target="_blank" style="background-color: orange; color: black; padding: 5px 10px; text-decoration: none; border-radius: 5px;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
