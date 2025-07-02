import numpy as np
from utils.activations import activation_functions

class Perceptron:
    def __init__(self, lr=0.01, epochs=1000, activation='step'):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation_functions[activation]

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                output = self.activation(z)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.activation(z) >= 0.5).astype(int)
