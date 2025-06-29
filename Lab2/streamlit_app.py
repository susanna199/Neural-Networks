import streamlit as st
import numpy as np
from models.perceptron import Perceptron
from models.adaline import Adaline
from models.madaline import Madaline
from utils.activations import activations_list
from utils.preprocess import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Title
st.title("Neural Network Simulator")
st.write("Train and test basic neural network models: Perceptron, Adaline, Madaline")

# Sidebar for model and parameters
model_type = st.sidebar.selectbox("Select Model", ["Perceptron", "ADALINE", "MADALINE"])
activation = st.sidebar.selectbox("Activation Function", activations_list)
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%f")
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=2000, value=100, step=10)

# Load dataset
X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

# Train model
if st.button("Train Model"):
    if model_type == "Perceptron":
        model = Perceptron(lr=lr, epochs=epochs, activation=activation)
    elif model_type == "ADALINE":
        model = Adaline(lr=lr, epochs=epochs)
    elif model_type == "MADALINE":
        model = Madaline(lr=lr, epochs=epochs, activation=activation)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_bin = np.where(y_test == -1, 0, 1)
    y_pred_bin = np.where(y_pred == -1, 0, 1)

    acc = accuracy_score(y_test_bin, y_pred_bin)
    report = classification_report(y_test_bin, y_pred_bin, output_dict=False)

    # Display results
    st.success(f"✅ Accuracy: {acc:.4f}")
    st.text("\nClassification Report:")
    st.code(report)

    # Plot MSE for Adaline
    if model_type == "ADALINE":
        st.write("### Mean Squared Error over Epochs")
        fig, ax = plt.subplots()
        ax.plot(model.cost)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("ADALINE Learning Curve")
        ax.grid(True)
        st.pyplot(fig)
