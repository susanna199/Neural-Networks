import numpy as np
np.random.seed(42)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Activation Functions
# -----------------------
def step(z): return np.where(z >= 0, 1, 0)
def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def linear(z): return z

def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def tanh_derivative(z): return 1 - np.tanh(z)**2
def relu_derivative(z): return np.where(z > 0, 1, 0)
def linear_derivative(z): return np.ones_like(z)

derivatives = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'linear': linear_derivative
}

activation_functions = {
    'step': step,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'linear': linear
}

# -----------------------
# Perceptron
# -----------------------
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000, activation='step'):
        self.lr = lr
        self.epochs = epochs
        self.activation_name = activation
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

# -----------------------
# Adaline
# -----------------------
class Adaline:
    def __init__(self, lr=0.01, epochs=1000, activation='linear'):
        self.lr = lr
        self.epochs = epochs
        self.activation_name = activation
        self.activation_fn = activation_functions[activation]
        self.activation_derivative = derivatives.get(activation, None)

    def fit(self, X, y):
        # if self.activation_name == "linear":
        #     y = np.where(y == 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                z = np.dot(X[i], self.weights) + self.bias
                output = self.activation_fn(z)
                if self.activation_derivative is not None:
                    grad = (y[i] - output) * self.activation_derivative(z)
                else:
                    grad = y[i] - output  # fallback
                self.weights += self.lr * grad * X[i]
                self.bias += self.lr * grad

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        output = self.activation_fn(z)
        # if self.activation_name == "linear":
        #     return np.where(output >= 0, 1, -1)
        # else:
        #     return np.where(output >= 0.5, 1, 0)
        return np.where(output >= 0.5, 1, 0)

# class Adaline:
#     def __init__(self, lr=0.01, epochs=1000):
#         self.lr = lr
#         self.epochs = epochs

#     def fit(self, X, y):
#         # For true Adaline, y should be in {0,1}, but MSE will still work
#         self.weights = np.zeros(X.shape[1])
#         self.bias = 0

#         for _ in range(self.epochs):
#             # Net input
#             z = np.dot(X, self.weights) + self.bias

#             # Output is linear (no activation function)
#             output = z

#             # Error (difference from actual target)
#             error = y - output

#             # Weight and bias updates using gradient of MSE
#             self.weights += self.lr * np.dot(X.T, error)
#             self.bias += self.lr * np.sum(error)

#     def predict(self, X):
#         z = np.dot(X, self.weights) + self.bias
#         return np.where(z >= 0.5, 1, 0)



# -----------------------
# Madaline (Ensemble of Adaline)
# -----------------------
# class Madaline:
#     def __init__(self, n_units=3, lr=0.01, epochs=1000, activation='linear'):
#         self.n_units = n_units
#         self.lr = lr
#         self.epochs = epochs
#         self.activation_name = activation
#         self.activation = activation_functions[activation]
#         self.units = [Adaline(lr=lr, epochs=epochs, activation=activation) for _ in range(n_units)]

#     def fit(self, X, y):
#         for unit in self.units:
#             unit.fit(X, y)

#     # def predict(self, X):
#     #     outputs = np.array([unit.activation_fn(np.dot(X, unit.weights) + unit.bias) for unit in self.units])
#     #     avg_output = np.mean(outputs, axis=0)
#     #     return (avg_output >= 0.5).astype(int)
#     def predict(self, X):
#         predictions = np.array([unit.predict(X) for unit in self.units])
#         majority_vote = np.round(np.mean(predictions, axis=0)).astype(int)
#         return majority_vote

class Madaline:
    def __init__(self, n_units=2, lr=0.01, epochs=1000, activation='step'):
        self.n_units = n_units
        self.lr = lr
        self.epochs = epochs
        self.activation_name = activation
        self.activation_fn = activation_functions[activation]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, 0, 1)  # Keep labels as 0/1 for compatibility

        self.w = np.random.uniform(-1, 1, (self.n_units, n_features))
        self.b = np.random.uniform(-1, 1, self.n_units)
        self.v = np.ones(self.n_units)  # Fixed output layer weights
        self.b_out = 0.0

        for epoch in range(self.epochs):
            errors = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]

                zin = np.dot(self.w, xi) + self.b
                z = self.activation_fn(zin)

                # If step, convert to binary {-1, 1}
                if self.activation_name == 'step':
                    z = z * 2 - 1

                yin = np.dot(self.v, z) + self.b_out
                y_pred = 1 if yin >= 0.5 else 0

                # Update if prediction is wrong
                if y_pred != target:
                    errors += 1
                    idx = np.argmin(np.abs(zin))
                    delta = (target - y_pred)
                    self.w[idx] += self.lr * delta * xi
                    self.b[idx] += self.lr * delta

            if errors == 0:
                break

    def predict(self, X):
        zin = np.dot(self.w, X.T).T + self.b
        z = self.activation_fn(zin)
        if self.activation_name == 'step':
            z = z * 2 - 1
        yin = np.dot(z, self.v) + self.b_out
        return np.where(yin >= 0.5, 1, 0)


# -----------------------
# Experiment Runner
# -----------------------
def run_model(model_name, activation):
    print(f"\n=== {model_name.upper()} | Activation: {activation.upper()} ===")

    if activation == 'step' and model_name in ['adaline', 'madaline']:
        print("⚠️ WARNING: 'step' activation is not suitable for Adaline/Madaline.")
        
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if model_name == 'perceptron':
        model = Perceptron(lr=0.01, epochs=1000, activation=activation)
    elif model_name == 'adaline':
        model = Adaline(lr=0.01, epochs=1000, activation=activation)
    elif model_name == 'madaline':
        model = Madaline(lr=0.01, epochs=1000, activation=activation)
    else:
        raise ValueError("Unknown model")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Normalize predictions for linear Adaline
    # if model_name == 'adaline' and activation == 'linear':
    #     y_test = np.where(y_test == 0, -1, 1)
    #     y_pred = np.where(y_pred == -1, 0, 1)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["malignant", "benign"]))

# Run all
models = ['perceptron', 'adaline', 'madaline']
activations = ['step', 'sigmoid', 'tanh', 'relu', 'linear']

for model in models:
    for act in activations:
        run_model(model, act)
