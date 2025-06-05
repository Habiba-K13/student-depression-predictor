import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.title("Student Depression Prediction (Top 2 Models)")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload the dataset 'student_depression_dataset.csv' to proceed.")
    st.stop()

# EDA
st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.text(f"Shape: {df.shape}")
st.write(df.describe())

if st.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

# Visualization
st.subheader("Depression Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Depression', data=df, ax=ax)
st.pyplot(fig)

# Encoding and Splitting
df = df.dropna()
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

X = df.drop("Depression", axis=1)
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
st.subheader("Choose a Model")
model_name = st.selectbox("Select model", ["Random Forest", "Logistic Regression"])

if st.button("Train and Evaluate"):
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

