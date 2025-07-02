import numpy as np
from utils.activations import activation_functions

class Madaline:
    def __init__(self, n_units=2, lr=0.01, epochs=1000, activation='step'):
        self.n_units = n_units
        self.lr = lr
        self.epochs = epochs
        self.activation_name = activation
        self.activation_fn = activation_functions[activation]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, 0, 1)

        self.w = np.random.uniform(-1, 1, (self.n_units, n_features))
        self.b = np.random.uniform(-1, 1, self.n_units)
        self.v = np.ones(self.n_units)
        self.b_out = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                xi, target = X[i], y[i]
                zin = np.dot(self.w, xi) + self.b
                z = self.activation_fn(zin)
                if self.activation_name == 'step':
                    z = z * 2 - 1
                yin = np.dot(self.v, z) + self.b_out
                y_pred = 1 if yin >= 0.5 else 0

                if y_pred != target:
                    idx = np.argmin(np.abs(zin))
                    delta = target - y_pred
                    self.w[idx] += self.lr * delta * xi
                    self.b[idx] += self.lr * delta

    def predict(self, X):
        zin = np.dot(self.w, X.T).T + self.b
        z = self.activation_fn(zin)
        if self.activation_name == 'step':
            z = z * 2 - 1
        yin = np.dot(z, self.v) + self.b_out
        return np.where(yin >= 0.5, 1, 0)
