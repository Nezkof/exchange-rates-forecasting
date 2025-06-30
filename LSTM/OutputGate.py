from LSTM.FCL import FCL

from helpers.useFunctions import use_tanh, use_sigmoid
from helpers.useMath import use_vector_multiplication

class OutputGate:
   def __init__(self, hidden_size):
      self.hidden_size = hidden_size
      self.sigmoid_layer = FCL(use_sigmoid, hidden_size)

   def compute(self, x):
      z_vector = self.h_prev + [x]
      tanh_c_values = [use_tanh(x) for x in self.c_prev]
      sigmoid_values = self.sigmoid_layer.calculate(z_vector, self.weights, self.biases)

      return [ self.c_prev, use_vector_multiplication(tanh_c_values,sigmoid_values) ]
   
   def set_biases(self, bias):
      self.biases = bias 
   
   def set_weights(self, weight):
      self.weights = weight   
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev 
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev 
