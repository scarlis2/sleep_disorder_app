import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


st.set_page_config(
    page_title="Sleep Disorder Screening App",
    layout="wide"
)

st.title("Wearable Data and Machine Learning for Early Detection of Sleep Disorders")

st.write(
    "This app trains a Random Forest model at runtime using uploaded wearable-simulated sleep data. "
    "No saved .pkl file is required."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your processed sleep feature CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a processed CSV file that includes feature columns and a target column such as stage, sleep_stage, label, or target.")
    st.stop()

data = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.write(f"Rows: {data.shape[0]}")
st.write(f"Columns: {data.shape[1]}")

st.subheader("Column Names")
st.write(list(data.columns))

# Try to automatically find the target column
target_names = ["stage", "sleep_stage", "label", "target", "class"]
possible_targets = [
    col for col in data.columns
    if col.lower().strip() in target_names
]

if possible_targets:
    default_index = list(data.columns).index(possible_targets[0])
else:
    default_index = 0

target_column = st.sidebar.selectbox(
    "Select the target column",
    data.columns,
    index=default_index
)

df = data.copy().dropna()

if df.empty:
    st.error("After removing missing values, no rows are left. Please upload a cleaner or larger CSV file.")
    st.stop()

X = df.drop(columns=[target_column])
y = df[target_column]

# Prevent using continuous numeric columns as target
target_type = type_of_target(y)

if target_type not in ["binary", "multiclass"]:
    st.error(
        f"The selected target column '{target_column}' does not look like a classification label. "
        "Please choose a column such as stage, sleep_stage, label, or target."
    )
    st.write("Detected target type:", target_type)
    st.stop()

if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
    st.error(
        f"The selected target column '{target_column}' appears to be a continuous numeric feature, not a sleep-stage label. "
        "Please select a true class label column such as stage."
    )
    st.stop()

# Remove non-useful columns from features
for col in X.columns:
    if X[col].nunique() <= 1:
        X = X.drop(columns=[col])

# Encode categorical feature columns
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Keep numeric features only
X = X.select_dtypes(include=[np.number])

if X.empty:
    st.error("No numeric feature columns were found after preprocessing. Please upload a processed feature dataset.")
    st.stop()

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))
class_names = label_encoder.classes_

class_counts = pd.Series(y.astype(str)).value_counts()

st.subheader("Target Class Distribution")
st.dataframe(class_counts.rename_axis("Class").reset_index(name="Count"))

if len(class_counts) < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()

if len(df) < 10:
    st.error("The dataset is too small to train a reliable model. Please upload a larger processed CSV file.")
    st.stop()

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

# Stratify only when every class has enough samples
if class_counts.min() >= 2:
    stratify_option = y_encoded
else:
    stratify_option = None
    st.warning(
        "Stratified splitting was turned off because one or more classes has fewer than 2 samples."
    )

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=stratify_option
    )
except ValueError as e:
    st.error(
        "The dataset could not be split into training and testing sets. "
        "Please upload a larger CSV file with more examples for each sleep stage."
    )
    st.write("Technical detail:", str(e))
    st.stop()

model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    class_weight="balanced"
)

try:
    model.fit(X_train, y_train)
except ValueError as e:
    st.error("The model could not train with the selected target column.")
    st.write("Please make sure your target column is a category such as W, N1, N2, N3, REM, or similar.")
    st.write("Technical detail:", str(e))
    st.stop()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.3f}")

with col2:
    st.metric("Weighted F1 Score", f"{f1:.3f}")

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[str(c) for c in class_names],
    output_dict=True,
    zero_division=0
)

st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)

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
top_features = importance_df.head(10)

ax2.barh(top_features["Feature"], top_features["Importance"])
ax2.set_title("Top 10 Feature Importances")
ax2.set_xlabel("Importance")
ax2.invert_yaxis()

st.pyplot(fig2)

st.success("Model trained successfully without using a .pkl file.")