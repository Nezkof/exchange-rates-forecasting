from helpers.functions import use_sigmoid, use_tanh
from helpers.useMath import use_linear_combination, use_vector_sum, use_vector_multiplication

class InputGate:
   def __init__(self, ):
      pass

   def calc_sigmoid(self, x):
      weights = self.weights[0]
      biases = self.biases[0]
      sigmoid_values = []
   
      args = self.h_prev + x
      linear_combination = use_linear_combination(weights,args)
      vector_sum = use_vector_sum(linear_combination, biases)

      for x in vector_sum:
         sigmoid_values.append(use_sigmoid(x))
      
      return sigmoid_values
   
   def calc_tanh(self, x):
      weights = self.weights[1]
      biases = self.biases[1]
      tanh_values = []
   
      args = self.h_prev + x
      linear_combination = use_linear_combination(weights,args)
      vector_sum = use_vector_sum(linear_combination,biases)

      for x in vector_sum:
         tanh_values.append(use_tanh(x))
      
      return tanh_values

   def compute(self, x):
      sigmoid_values = self.calc_sigmoid(x)
      tanh_values = self.calc_tanh(x)

      return (
         use_vector_sum(
            use_vector_multiplication(tanh_values , sigmoid_values),
            self.c_prev)
      )

   def set_bias(self, bias):
      self.biases = bias # [ [sigm_bias], [tanh_bias] ]
   
   def set_weight(self, weight):
      self.weights = weight # [ [sigm_w1, sigm_w2] [tanh_w1, tanh_w2] ]
   
   def set_c_prev(self, c_prev):
      self.c_prev = c_prev # [c_prev]
   
   def set_h_prev(self, h_prev):
      self.h_prev = h_prev # [h_prev]
