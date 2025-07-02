import numpy as np
np.random.seed(42)
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from models import perceptron, adaline, madaline
from utils.activations import activation_functions
from utils.preprocess import load_default_dataset, load_uploaded_dataset

st.title("Neural Network Simulator")
st.markdown("Train **Perceptron**, **Adaline**, or **Madaline** on any dataset")
st.markdown("*Default dataset: [Breast Cancer Diagnostic Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from scikit-learn*")
# ---------------------
# Sidebar Configuration
# ---------------------
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Model", ["Perceptron", "Adaline", "Madaline"])
activation = st.sidebar.selectbox("Activation", list(activation_functions.keys()))
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
epochs = st.sidebar.slider("Epochs", 100, 3000, step=100, value=1000)
use_upload = st.sidebar.checkbox("Upload Custom Dataset")

if model_name == "Madaline":
    n_units = st.sidebar.slider("Hidden Units", 1, 10, value=2)
else:
    n_units = None

# ---------------------
# Load Dataset
# ---------------------
if use_upload:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview", df.head())
        target_column = st.selectbox("Select Target Column", df.columns)
        
        uploaded_file.seek(0)
        try:
            X_train, X_test, y_train, y_test = load_uploaded_dataset(uploaded_file, target_column)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.warning("Upload a dataset to continue.")
        st.stop()
else:
    X_train, X_test, y_train, y_test = load_default_dataset()

# ---------------------
# Model Training
# ---------------------
if st.button("Train Model"):
    if model_name == "Perceptron":
        model = perceptron.Perceptron(lr=lr, epochs=epochs, activation=activation)
    elif model_name == "Adaline":
        model = adaline.Adaline(lr=lr, epochs=epochs, activation=activation)
    elif model_name == "Madaline":
        model = madaline.Madaline(n_units=n_units, lr=lr, epochs=epochs, activation=activation)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.success(f"✅ Model trained! Accuracy: **{acc:.4f}**")

    # ---------------------
    # Classification Report
    # ---------------------
    st.markdown("### Classification Report")
    classes = [k for k in report.keys() if k in ['0', '1'] or isinstance(report[k], dict)]
    display_report = {
        "Precision": {k: f"{v['precision']:.2f}" for k, v in report.items() if k in classes},
        "Recall":    {k: f"{v['recall']:.2f}"    for k, v in report.items() if k in classes},
        "F1-Score":  {k: f"{v['f1-score']:.2f}"  for k, v in report.items() if k in classes}
    }

    st.table(display_report)
