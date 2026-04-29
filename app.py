import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Sleep Disorder Screening App",
    layout="wide"
)

st.title("Wearable Data and Machine Learning for Early Detection of Sleep Disorders")

st.write(
    "This app trains a machine learning model at runtime using wearable-simulated sleep data. "
    "It does not require a saved .pkl model file."
)

st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload your processed sleep feature CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Dataset Summary")
    st.write(data.describe(include="all"))

    st.sidebar.header("Model Settings")

    target_column = st.sidebar.selectbox(
        "Select the target column",
        data.columns
    )

    if target_column:
        df = data.copy()

        df = df.dropna()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical feature columns
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Encode target column if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y.astype(str))

        test_size = st.sidebar.slider(
            "Test size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05
        )

        n_estimators = st.sidebar.slider(
            "Number of trees",
            min_value=50,
            max_value=300,
            value=100,
            step=50
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")

        with col2:
            st.metric("Weighted F1 Score", f"{f1:.3f}")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

        st.subheader("Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)

        fig2, ax2 = plt.subplots()
        ax2.barh(
            importance_df["Feature"].head(10),
            importance_df["Importance"].head(10)
        )
        ax2.set_title("Top 10 Feature Importances")
        ax2.set_xlabel("Importance")
        ax2.invert_yaxis()

        st.pyplot(fig2)

        st.success("Model trained successfully without using a .pkl file.")

else:
    st.info("Upload your processed CSV file to train the model.")