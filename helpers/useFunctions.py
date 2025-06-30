import numpy as np

def use_sigmoid(x):
   return 1/(1+np.exp(-x))


def use_tanh(x):
   return np.tanh(x)
