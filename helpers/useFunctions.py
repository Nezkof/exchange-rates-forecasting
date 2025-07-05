import math
import numpy as np

def use_sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def use_tanh(x):
    return float(np.tanh(x))

def use_identity(x):
    return x

def rastrigin(x):
   A = 10
   n = len(x)
   return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])

def styblinski_tang(x):
   return 0.5 * sum([xi**4 - 16*xi**2 + 5*xi for xi in x])

def holder_table(x_array):
   x = x_array[0]
   y = x_array[1]
   term1 = math.sin(x)
   term2 = math.cos(y)
   term3 = math.exp(abs(1 - math.sqrt(x**2 + y**2) / math.pi))
   return -abs(term1 * term2 * term3)

def inverse_proportional(max, x):
   return max / (x + 1)

def gauss(a, c, x):
   exponent = -a * (x - c)**2
   if exponent < -700 or exponent > 700:   
      return 0.0
   return math.exp(exponent)

def parabola(x):
   # return x**2
   return x