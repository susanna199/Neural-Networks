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
