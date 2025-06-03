import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Page configuration
st.set_page_config(page_title="Student Depression Classifier", layout="wide")
st.title("🎓 Student Depression Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your student_depression_dataset.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    if 'Depression' not in df.columns:
        st.error("❌ Dataset must contain a 'Depression' column.")
    else:
        # Split features and label
        X = df.drop("Depression", axis=1)
        y = df["Depression"]

        # Replace missing values and convert
        X = X.replace('?', np.nan)
        X = X.apply(pd.to_numeric, errors='coerce')
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.write(f"🧠 Training set size: {X_train.shape}")
        st.write(f"🧪 Testing set size: {X_test.shape}")

        # ---------------------------------------------
        # Model Selection
        # ---------------------------------------------
        model_choice = st.selectbox("🔍 Choose a Classification Model", ["Random Forest", "Logistic Regression"])

        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------------------------------------------
        # Evaluation
        # ---------------------------------------------
        st.subheader(f"📊 {model_choice} Evaluation")

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        accuracy = report["accuracy"]
        st.success(f"✅ Accuracy of {model_choice}: {accuracy:.2%}")

        # Feature importance (only for Random Forest)
        if model_choice == "Random Forest":
            st.subheader("🔍 Top 10 Feature Importances")
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=feature_imp.head(10), y=feature_imp.head(10).index, ax=ax)
            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin.")
