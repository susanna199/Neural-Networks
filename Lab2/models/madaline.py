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
