from helpers.functions import use_tanh, use_sigmoid
from helpers.useMath import use_vector_sum, use_linear_combination, use_vector_multiplication

class OutputGate:
   def __init__(self, ):
      pass

   def calc_tanh(self):
      tanh_values = []

      for c in self.c_prev:
         tanh_values.append(use_tanh(c))
      
      return tanh_values

   def calc_sigmoid(self, x):
      sigmoid_values = []

      args = self.h_prev + x
      linear_combination = use_linear_combination(self.weights, args)
      vector_sum = use_vector_sum(linear_combination, self.biases)
      for x in vector_sum:
         sigmoid_values.append(use_sigmoid(x))

      return sigmoid_values

   def compute(self, x):
      tanh_values = self.calc_tanh()
      sigmoid_values = self.calc_sigmoid(x)
      h = use_vector_multiplication(sigmoid_values, tanh_values)

      return [self.c_prev, h]
   
   def set_bias(self, bias):
      self.biases = bias  # [bias]
   
   def set_weight(self, weight):
      self.weights = weight  # [ 
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev # [c_prev]
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev # [h_prev]
