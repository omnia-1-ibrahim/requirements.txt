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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

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
data_url = "https://raw.githubusercontent.com/omnia-1-ibrahim/requirements.txt/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # استبدل الرابط برابط الـ GitHub الخاص بك
data = pd.read_csv(data_url)

# الصفحة الرئيسية
st.title("Telecom Churn Analysis and Prediction")
st.image("dataset-cover.png", use_column_width=True)  # تأكد من تحميل الصورة

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
if 'SomeCategoryColumn' in data.columns:  # استبدل 'SomeCategoryColumn' بالعمود المناسب
    fig = px.histogram(data, x='Churn', color='SomeCategoryColumn', title='Churn Distribution by Category')
    st.plotly_chart(fig)

# تحليل ارتباط البيانات
st.subheader('Correlation Heatmap')
plt.figure(figsize=(10, 6))

# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])

# Check if there are enough numeric columns to compute correlation
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

X = data.drop('Churn', axis=1)  # استبدل 'Churn' بالعمود المستهدف
y = data['Churn']  # استبدل 'Churn' بالعمود المستهدف
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# اختيار النموذج
model_choice = st.selectbox("Choose a model for prediction", ("Random Forest", "Logistic Regression", "XGBoost"))

if model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = XGBClassifier()

model.fit(X_train, y_train)

# نموذج جاهز للتنبؤ
st.write("Model is ready for prediction.")
user_input = st.text_input('Enter customer features as comma-separated values (matching the dataset columns)')

if st.button('Predict'):
    try:
        input_data = [float(x) for x in user_input.split(',')]
        if len(input_data) == X.shape[1]:  # Check if the input length matches the number of features
            prediction = model.predict([input_data])
            st.write(f'The customer is predicted to {"churn" if prediction[0] == 1 else "not churn"}.')
        else:
            st.error(f"Please enter {X.shape[1]} values.")
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")



# عرض الموارد
st.subheader("View Resources")

st.subheader("Jupyter Notebook")
st.write("You can view the Jupyter Notebook [here](https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/final_with_mlflow%20(1).ipynb)")  # استبدل بالرابط الصحيح

st.subheader("Power BI Report")
st.write("Check the Power BI report embedded below:")
st.markdown(
    '<a href="file:https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/graduation%20project.pptx" target="_blank">Open Presentation</a>',
    unsafe_allow_html=True
)

st.subheader("Presentation")
st.write("Upload your presentation file below:")
uploaded_presentation = st.file_uploader("Upload your PowerPoint presentation", type=["pptx", "ppt"])

if uploaded_presentation:
    st.success("Presentation uploaded successfully!")
    st.write("You can view it in PowerPoint or other presentation software.")

# عرض معلومات الفريق
st.subheader("Meet Our Team")

team_members = [
    {
        "name": "Omnia Ibrahim Sayed",
        "linkedin": "https://www.linkedin.com/in/omnia-ibrahim-8168b022b"  # رابط LinkedIn الخاص بك
    },
    {
        "name": "Yossef Mohamed Mohamed",
        "linkedin": "https://www.linkedin.com/in/yossef-mohamed/"
    },
    {
        "name": "Abdelrahman Sherif Kamel",
        "linkedin": "https://www.linkedin.com/in/abdelrahman-kamel/"
    }
]

for member in team_members:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="margin-right: 10px; font-size: 16px;">{member['name']}</span>
            <a href="{member['linkedin']}" target="_blank" style="background-color: orange; color: black; padding: 5px 10px; text-decoration: none; border-radius: 5px;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
