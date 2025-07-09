import numpy as np

from helpers.useFunctions import sigmoid

from helpers.useRandom import random_array

class ForgetGate: 
   def __init__(self, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      
      hx_length = hidden_size + features_number
      self.f_weights = random_array(-0.1, 0.1, hidden_size, hx_length)
      self.f_biases = random_array(-0.1, 0.1, hidden_size)
      self.f_weights_derivative = np.zeros((hidden_size,hx_length))
      self.f_biases_derivative = np.zeros(hidden_size)

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []
      
      self.f_output = []

   def forward(self, x, c_prev = None, h_prev = None):
      if (c_prev is not None and h_prev is not None):
         self.c_prev = c_prev
         self.h_prev = h_prev
      
      self.xc = np.hstack((x, self.h_prev))
      self.f_output = sigmoid(np.dot(self.f_weights, self.xc) + self.f_biases)
      return self.c_prev, self.f_output
      

      

