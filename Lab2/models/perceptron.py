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

