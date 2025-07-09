import numpy as np

from helpers.useFunctions import sigmoid, tanh
from helpers.useRandom import random_array

class InputGate:
   def __init__(self, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      hx_length = hidden_size + features_number

      self.i_weights = random_array(-0.1, 0.1, hidden_size, hx_length)
      self.c_weights = random_array(-0.1, 0.1, hidden_size, hx_length)
      self.i_biases = random_array(-0.1, 0.1, hidden_size)
      self.c_biases = random_array(-0.1, 0.1, hidden_size)

      self.i_weights_derivative = np.zeros((hidden_size,hx_length))
      self.c_weights_derivative = np.zeros((hidden_size,hx_length))
      self.i_biases_derivative = np.zeros(hidden_size)
      self.c_biases_derivative = np.zeros(hidden_size)

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []

   def forward(self, x, h_prev = None):
      if (h_prev is not None):
         self.h_prev = h_prev
   
      self.xc = np.hstack((x, self.h_prev))
      self.i_output = sigmoid(np.dot(self.i_weights, self.xc) + self.i_biases)
      self.c_output = tanh(np.dot(self.c_weights, self.xc) + self.c_biases)

      return self.i_output, self.c_output

   def get_c_output(self):
      return self.c_output
   
   def get_i_output(self):
      return self.i_output