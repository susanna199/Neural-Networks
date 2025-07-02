import numpy as np

def step(z): return np.where(z >= 0, 1, 0)
def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def linear(z): return z

def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
def tanh_derivative(z): return 1 - np.tanh(z)**2
def relu_derivative(z): return np.where(z > 0, 1, 0)
def linear_derivative(z): return np.ones_like(z)

activation_functions = {
    'step': step,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'linear': linear
}

derivatives = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'linear': linear_derivative
}
