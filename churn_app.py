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

# Application title
st.title("Professional Telecom Churn Analysis and Prediction")

# Load default dataset option
if st.checkbox("Use example dataset"):
    data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')  # Replace with an actual churn dataset
    st.write("Using example dataset.")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file or st.checkbox("Use example dataset"):
    if not st.checkbox("Use example dataset"):
        data = pd.read_csv(uploaded_file)

    # Display raw data
    st.subheader('Raw Data')
    st.write(data.head())

    # Data summary
    st.subheader('Dataset Summary')
    st.write(data.describe())

    # Feature selection
    st.subheader('Feature Selection')
    selected_features = st.multiselect('Select features for analysis', data.columns.tolist(), default=data.columns.tolist())

    # Target selection
    target_column = st.selectbox("Select target variable", data.columns.tolist(), index=data.columns.tolist().index('Churn'))  # Replace 'Churn' with actual target column

    # Exploratory data analysis
    st.subheader('Exploratory Data Analysis')
    if st.checkbox('Show distribution of target variable'):
        churn_count = data[target_column].value_counts()
        st.write(f"Churned: {churn_count[1]}, Not churned: {churn_count[0]}")
        fig = px.histogram(data, x=target_column, title='Churn Distribution')
        st.plotly_chart(fig)

    # Data Preparation
    st.subheader('Data Preparation')
    # Convert categorical columns to numeric
    label_encoder = LabelEncoder()
    for column in selected_features:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Split the data into features (X) and target (y)
    X = data[selected_features]
    y = data[target_column]

    # Split ratio input
    test_size = st.slider('Select test data size (in %)', 10, 50, 20) / 100

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model selection
    st.subheader('Model Selection and Training')
    model_choice = st.selectbox("Choose a model", ('Random Forest', 'Logistic Regression', 'XGBoost'))

    n_estimators = st.slider('Number of trees for Random Forest', 100, 1000, step=50)
    learning_rate = st.slider('Learning rate for XGBoost', 0.01, 0.3, step=0.01)

    if model_choice == 'Random Forest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    else:
        model = XGBClassifier(learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    st.subheader('Model Performance')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f'Accuracy: {accuracy * 100:.2f}%')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')

    # Display classification report
    st.subheader('Classification Report')
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(plt)

    # Feature Importance (only for Random Forest)
    if model_choice == 'Random Forest':
        st.subheader('Feature Importance')
        importance = pd.Series(model.feature_importances_, index=selected_features)
        importance = importance.sort_values(ascending=False)
        sns.barplot(x=importance, y=importance.index)
        st.pyplot(plt)

    # Real-Time Prediction
    st.subheader('Make a Real-Time Prediction')
    user_input = st.text_input('Enter customer features as comma-separated values')

    if st.button('Predict'):
        # Convert input into dataframe
        input_data = [float(x) for x in user_input.split(',')]
        input_df = pd.DataFrame([input_data], columns=selected_features)

        # Make prediction
        prediction = model.predict(input_df)
        st.write(f'The customer is predicted to {"churn" if prediction[0] == 1 else "not churn"}.')

    # Save the trained model
    if st.button('Save Model'):
        joblib.dump(model, 'churn_model.pkl')
        st.write("Model saved successfully!")

    # Recommendations
    st.subheader('Recommendations Based on Analysis')
    st.write("""
    1. New customers are more likely to churn. Offer attractive deals to retain them.
    2. Provide rewards or gifts for customers with short-term contracts to encourage renewal.
    3. Long-tenure customers rarely churn, so offer them privileges to maintain loyalty.
    """)

# If no file is uploaded
else:
    st.write("Please upload a CSV file or use the example dataset to start the analysis.")
