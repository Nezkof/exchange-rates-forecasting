import numpy as np

from helpers.useFunctions import sigmoid, tanh
from helpers.useMath import use_vector_multiplication
from helpers.useRandom import random_array

class OutputGate:
   def __init__(self, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      hx_length = hidden_size + features_number

      self.o_weights = random_array(-0.1, 0.1, hidden_size, hx_length)
      self.o_biases = random_array(-0.1, 0.1, hidden_size)
      self.o_weights_derivative = np.zeros((hidden_size,hx_length))
      self.o_biases_derivative = np.zeros(hidden_size)

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []


   def forward(self, x,  h_prev = None):
      if (h_prev is not None):
         self.h_prev = h_prev

      self.xc = np.hstack((x, self.h_prev))
      self.o_output = sigmoid(np.dot(self.o_weights, self.xc) + self.o_biases)

      return self.o_output

   def get_o_output(self):
      return self.o_output
   
   def get_c_prev(self):
      return self.c_prev