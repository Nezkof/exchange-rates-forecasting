from FCL import FCL

from helpers.useFunctions import use_sigmoid
from helpers.useMath import use_vector_multiplication

class ForgetGate: 
   def __init__(self, hidden_size):
      self.hidden_size = hidden_size
      self.sigmoid_layer = FCL(use_sigmoid, hidden_size)
   
   def compute(self, x):
      z_vector = self.h_prev + x
      sigmoid_values = self.sigmoid_layer.calculate(z_vector, self.weights, self.biases)
      return use_vector_multiplication(self.c_prev, sigmoid_values)
   
   def set_biases(self, biases):
      self.biases = biases
   
   def set_weights(self, weights):
      self.weights = weights
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev

