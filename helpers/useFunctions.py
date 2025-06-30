import numpy as np

def use_sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def use_tanh(x):
    return float(np.tanh(x))

def use_identity(x):
    return x