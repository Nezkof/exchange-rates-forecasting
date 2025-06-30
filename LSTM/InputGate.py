from LSTM.FCL import FCL

from helpers.useFunctions import use_sigmoid, use_tanh
from helpers.useMath import use_vector_sum, use_vector_multiplication

class InputGate:
   def __init__(self, hidden_size):
      self.hidden_size = hidden_size
      self.sigmoid_layer = FCL(use_sigmoid, hidden_size)
      self.tanh_layer = FCL(use_tanh, hidden_size)

   def compute(self, x):
      z_vector = self.h_prev + x
      sigmoid_values = self.sigmoid_layer.calculate(z_vector, self.sigmoid_weights, self.sigmoid_biases)
      tanh_values = self.tanh_layer.calculate(z_vector, self.tanh_weights, self.tanh_biases)

      return (
         use_vector_sum(
            use_vector_multiplication(tanh_values , sigmoid_values),
            self.c_prev)
      )
   
   def set_sigmoid_weights(self, weights):
      self.sigmoid_weights = weights
   
   def set_tanh_weights(self, weights):
      self.tanh_weights = weights

   def set_sigmoid_biases(self, biases):
      self.sigmoid_biases = biases
   
   def set_tanh_biases(self, biases):
      self.tanh_biases = biases
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev # [c_prev]
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev # [h_prev]
