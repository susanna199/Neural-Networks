# main.py - Neural Network Simulator

import numpy as np
from models.perceptron import Perceptron
from models.adaline import Adaline
from models.madaline import Madaline
from utils.activations import activations_list
from utils.preprocess import load_dataset
from sklearn.metrics import classification_report, accuracy_score


def menu():
    print("\n=== Neural Network Simulator ===")
    print("1. Perceptron")
    print("2. ADALINE")
    print("3. MADALINE")
    print("0. Exit")
    return input("Choose model: ")


def activation_menu():
    print("\n--- Activation Functions ---")
    for i, act in enumerate(activations_list):
        print(f"{i + 1}. {act}")
    return activations_list[int(input("Choose activation: ")) - 1]


def main():
    while True:
        choice = menu()
        if choice == '0':
            break

        X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

        act = activation_menu()
        lr = float(input("Learning Rate: "))
        epochs = int(input("Epochs: "))

        if choice == '1':
            model = Perceptron(lr=lr, epochs=epochs, activation=act)
        elif choice == '2':
            model = Adaline(lr=lr, epochs=epochs)
        elif choice == '3':
            model = Madaline(lr=lr, epochs=epochs, activation=act)
        else:
            print("Invalid choice!")
            continue

        print("\nTraining...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_binary = np.where(y_test == -1, 0, 1)
        y_pred_binary = np.where(y_pred == -1, 0, 1)

        print("\nEvaluation:")
        print("Accuracy:", accuracy_score(y_test_binary, y_pred_binary))
        print(classification_report(y_test_binary, y_pred_binary))


if __name__ == '__main__':
    main()


# models/perceptron.py
import numpy as np
from utils.activations import activation_fn, activation_derivative

class Perceptron:
    def __init__(self, lr=0.01, epochs=1000, activation='step'):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = activation_fn(z, self.activation)
                error = y[i] - y_pred

                if self.activation == 'step':
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error
                else:
                    dz = error * activation_derivative(z, self.activation)
                    self.weights += self.lr * dz * X[i]
                    self.bias += self.lr * dz

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        output = activation_fn(z, self.activation)
        return np.where(output >= 0, 1, -1)


# models/adaline.py
import numpy as np

class Adaline:
    def __init__(self, lr=0.0001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.cost = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            z = self.net_input(X)
            error = y - z
            self.weights += self.lr * X.T.dot(error) / n_samples
            self.bias += self.lr * error.mean()
            self.cost.append((error ** 2).mean())

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


# models/madaline.py
import numpy as np
from utils.activations import activation_fn

class Madaline:
    def __init__(self, lr=0.01, epochs=1000, activation='step'):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation
        self.hidden_units = 3

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.hidden_units, n_features))
        self.biases = np.zeros(self.hidden_units)
        self.output_weights = np.ones(self.hidden_units)

        for _ in range(self.epochs):
            for i in range(n_samples):
                net_inputs = np.dot(self.weights, X[i]) + self.biases
                activations = activation_fn(net_inputs, self.activation)
                output = np.sign(np.dot(self.output_weights, activations))
                error = y[i] - output

                if error != 0:
                    for j in range(self.hidden_units):
                        self.weights[j] += self.lr * error * X[i]
                        self.biases[j] += self.lr * error

    def predict(self, X):
        outputs = []
        for x in X:
            net_inputs = np.dot(self.weights, x) + self.biases
            activations = activation_fn(net_inputs, self.activation)
            output = np.sign(np.dot(self.output_weights, activations))
            outputs.append(output)
        return np.array(outputs)


# utils/activations.py
import numpy as np

def step(z):
    return np.where(z >= 0, 1, -1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu_derivative(z):
    return np.where(z > 0, 1, 0.01)

def activation_fn(z, func):
    return {
        'step': step,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        'leaky_relu': leaky_relu
    }[func](z)

def activation_derivative(z, func):
    return {
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'relu': relu_derivative,
        'leaky_relu': leaky_relu_derivative
    }.get(func, lambda z: 1)(z)

activations_list = ['step', 'sigmoid', 'tanh', 'relu', 'leaky_relu']


# utils/preprocess.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_dataset(name="breast_cancer"):
    if name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = np.where(data.target == 0, -1, 1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
