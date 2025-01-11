import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App
st.title("Interactive Machine Learning Classifier")
st.write("This app trains a classifier on the selected dataset and displays the evaluation metrics.")
# 1. File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features and target
    st.sidebar.header("Model Inputs")
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 2. Select Model
        st.sidebar.header("Select Classifier")
        classifier_name = st.sidebar.selectbox("Classifier", ["Random Forest", "SVM"])

        # Hyperparameters
        if classifier_name == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 100, step=10)
            max_depth = st.sidebar.slider("Max Depth", 2, 20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif classifier_name == "SVM":
            C = st.sidebar.slider("Regularization (C)", 0.01, 10.0)
            kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel, probability=True)

        # Train Model
        if st.sidebar.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if classifier_name == "Random Forest" else model.decision_function(X_test)

            # 3. Evaluation Metrics
            st.subheader("Model Evaluation")

            # Classification Report as DataFrame
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("**Classification Report**")
            st.dataframe(report_df)

            # Confusion Matrix as DataFrame
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df)

            # ROC Curve
            st.write("**ROC Curve**")
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)
