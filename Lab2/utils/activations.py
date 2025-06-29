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
