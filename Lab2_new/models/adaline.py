import numpy as np
from utils.activations import activation_functions, derivatives

class Adaline:
    def __init__(self, lr=0.01, epochs=1000, activation='linear'):
        self.lr = lr
        self.epochs = epochs
        self.activation_fn = activation_functions[activation]
        self.activation_derivative = derivatives.get(activation, None)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                z = np.dot(X[i], self.weights) + self.bias
                output = self.activation_fn(z)
                grad = (y[i] - output)
                if self.activation_derivative:
                    grad *= self.activation_derivative(z)
                self.weights += self.lr * grad * X[i]
                self.bias += self.lr * grad

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        output = self.activation_fn(z)
        return np.where(output >= 0.5, 1, 0)
