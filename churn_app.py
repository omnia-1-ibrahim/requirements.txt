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

# تحميل مجموعة البيانات من GitHub
data_url = "https://raw.githubusercontent.com/omnia-1-ibrahim/requirements.txt/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(data_url)

# الصفحة الرئيسية
st.title("Telecom Churn Analysis and Prediction")
st.image("dataset-cover.png", use_column_width=True)

# عرض البيانات الخام
st.subheader('Raw Data')
st.write(data.head())

# استعراض بيانات التحليل
st.subheader('Data Summary')
st.write(data.describe())

# تحليل البيانات باستخدام الرسوم البيانية
st.subheader('Churn Analysis')
churn_count = data['Churn'].value_counts()
st.bar_chart(churn_count)

# إضافة رسوم بيانية تفاعلية
st.subheader('Churn Distribution by Category')
if 'SomeCategoryColumn' in data.columns:
    fig = px.histogram(data, x='Churn', color='SomeCategoryColumn', title='Churn Distribution by Category')
    st.plotly_chart(fig)

# تحليل ارتباط البيانات
st.subheader('Correlation Heatmap')
plt.figure(figsize=(10, 6))
numeric_data = data.select_dtypes(include=[np.number])
if numeric_data.shape[1] > 1:
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
else:
    st.write("Not enough numeric columns to compute correlation.")
st.pyplot()

# إعداد نموذج التنبؤ
st.subheader("Churn Prediction")

label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إعداد النماذج
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# تدريب النماذج وحساب الأداء
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[model_name] = preds
    
    # حساب مقاييس الأداء
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    conf_matrix = confusion_matrix(y_test, preds)
    
    # عرض النتائج
    st.write(f"### {model_name} Performance:")
    st.write(f"*Accuracy:* {accuracy:.2f}")
    st.write(f"*Recall:* {recall:.2f}")
    st.write(f"*F1 Score:* {f1:.2f}")
    
    # عرض مصفوفة الارتباك
    st.write("*Confusion Matrix:*")
    st.write(conf_matrix)
    
    # رسم مصفوفة الارتباك
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    st.pyplot(plt)

# إدخال بيانات جديدة للتنبؤ
st.subheader("Predict Customer Churn")
input_data = {}

# إضافة حقول الإدخال للمدخلات الجديدة
input_data['gender'] = st.selectbox("Gender", ["Male", "Female"])
input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
input_data['Partner'] = st.selectbox("Partner", ["Yes", "No"])
input_data['Dependents'] = st.selectbox("Dependents", ["Yes", "No"])
input_data['tenure'] = st.number_input("Tenure (months)", min_value=0)
input_data['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"])
input_data['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
input_data['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
input_data['OnlineSecurity'] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
input_data['OnlineBackup'] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
input_data['DeviceProtection'] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
input_data['TechSupport'] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
input_data['StreamingTV'] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
input_data['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"])
input_data['PaymentMethod'] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
input_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    
    # Apply label encoding for each column in input data
    for column in input_df.select_dtypes(include=['object']).columns:
        input_df[column] = label_encoder.fit_transform(input_df[column])
        
    predictions = {}
    for model_name, model in models.items():
        preds = model.predict(input_df)
        predictions[model_name] = preds[0]
    
    st.write("Churn Prediction Results:")
    for model_name, prediction in predictions.items():
        st.write(f"{model_name}: **{'Yes' if prediction == 1 else 'No'}**")

# عرض الموارد
st.subheader("View Resources")
st.write("You can view the Jupyter Notebook [here](https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/final_with_mlflow%20(1).ipynb)")
st.write("You can view the Presentation [here](https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/graduation%20project.pptx)")
st.write("You can view the Power Bi [here](https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/4_6041671993533667237.pbix)")  

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

