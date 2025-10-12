import math
import numpy as np

def sigmoid(x):
   x = np.clip(x, -500, 500)  
   return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x): 
   return x*(1-x)

def tanh_derivative(x): 
   return 1.0 - x ** 2

def tanh(x):
   return np.tanh(x)

def inverse_proportional(max, x):
   return max / (x + 1)

def gauss(a, c, x):
   exponent = -a * (x - c)**2
   if exponent < -700 or exponent > 700:   
      return 0.0
   return math.exp(exponent)

def parabola(x):
   return x**2