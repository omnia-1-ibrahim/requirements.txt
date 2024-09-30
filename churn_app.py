import streamlit as st
import pandas as pd
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
data_url = "https://github.com/omnia-1-ibrahim/requirements.txt/blob/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # استبدل الرابط برابط الـ GitHub الخاص بك
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
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
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
    input_data = [float(x) for x in user_input.split(',')]
    prediction = model.predict([input_data])
    st.write(f'The customer is predicted to {"churn" if prediction[0] == 1 else "not churn"}.')

# عرض معلومات الفريق
st.subheader("Meet Our Team")
st.write("""
Our team consists of dedicated professionals:
- Abdelrahman Sherif Kamel
- Yossef Mohamed Mohamed
- Omnia Ibrahim Sayed
""")
st.write("Contact: support@example.com")  # إضافة معلومات الاتصال
st.image("path_to_team_image.jpg", use_column_width=True)  # استبدل بمسار صورة الفريق

# عرض الموارد
st.subheader("View Resources")

st.subheader("Jupyter Notebook")
st.write("You can view the Jupyter Notebook [here](https://nbviewer.jupyter.org/)")  # استبدل بالرابط الصحيح

st.subheader("Power BI Report")
st.write("Check the Power BI report embedded below:")
st.markdown(
    '<iframe width="800" height="600" src="YOUR_POWER_BI_EMBED_LINK" frameborder="0" allowFullScreen="true"></iframe>',
    unsafe_allow_html=True
)

st.subheader("Presentation")
st.write("Upload your presentation file below:")
uploaded_presentation = st.file_uploader("Upload your PowerPoint presentation", type=["pptx", "ppt"])

if uploaded_presentation:
    st.success("Presentation uploaded successfully!")
    st.write("You can view it in PowerPoint or other presentation software.")
