import tkinter as tk
from tkinter import ttk, messagebox
from models.perceptron import Perceptron
from models.adaline import Adaline
from models.madaline import Madaline
from utils.activations import activations_list
from utils.preprocess import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Simulator")
        self.create_widgets()

    def create_widgets(self):
        # Dropdowns for model and activation
        tk.Label(self.root, text="Select Model:").grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value="Perceptron")
        model_menu = ttk.Combobox(self.root, textvariable=self.model_var, values=["Perceptron", "ADALINE", "MADALINE"], state="readonly")
        model_menu.grid(row=0, column=1)

        tk.Label(self.root, text="Activation Function:").grid(row=1, column=0, sticky='w')
        self.activation_var = tk.StringVar(value=activations_list[0])
        activation_menu = ttk.Combobox(self.root, textvariable=self.activation_var, values=activations_list, state="readonly")
        activation_menu.grid(row=1, column=1)

        # Inputs for learning rate and epochs
        tk.Label(self.root, text="Learning Rate:").grid(row=2, column=0, sticky='w')
        self.lr_entry = tk.Entry(self.root)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=2, column=1)

        tk.Label(self.root, text="Epochs:").grid(row=3, column=0, sticky='w')
        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=3, column=1)

        # Button to start training
        tk.Button(self.root, text="Train Model", command=self.train_model).grid(row=4, column=0, columnspan=2, pady=10)

        # Output box
        self.output = tk.Text(self.root, height=20, width=70)
        self.output.grid(row=5, column=0, columnspan=2)

    def train_model(self):
        try:
            model_name = self.model_var.get()
            act = self.activation_var.get()
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

            if model_name == "Perceptron":
                model = Perceptron(lr=lr, epochs=epochs, activation=act)
            elif model_name == "ADALINE":
                model = Adaline(lr=lr, epochs=epochs)
            elif model_name == "MADALINE":
                model = Madaline(lr=lr, epochs=epochs, activation=act)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_test_binary = np.where(y_test == -1, 0, 1)
            y_pred_binary = np.where(y_pred == -1, 0, 1)

            acc = accuracy_score(y_test_binary, y_pred_binary)
            report = classification_report(y_test_binary, y_pred_binary)

            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Model: {model_name}\nActivation: {act}\n")
            self.output.insert(tk.END, f"Accuracy: {acc:.4f}\n\n{report}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()
