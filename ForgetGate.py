from helpers.functions import use_sigmoid
from helpers.useMath import use_linear_combination, use_vector_sum, use_vector_multiplication

class ForgetGate: 
   def __init__(self, ):
      pass
   
   def compute(self, x):
      sigmoid_values = []
      args = self.h_prev + x

      linear_combination = use_linear_combination(self.weights, args)
      vector_sum = use_vector_sum(linear_combination, self.bias)
      for x in vector_sum:
         sigmoid_values.append(use_sigmoid(x))

      return use_vector_multiplication(self.c_prev, sigmoid_values)
   
   def set_bias(self, bias):
      self.bias = bias
   
   def set_weight(self, weights):
      self.weights = weights
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev

