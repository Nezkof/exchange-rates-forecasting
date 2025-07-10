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

   def backward(self, derivative_i_output, derivative_c_tilda_output, di, dg, learning_rate):
      di_input = derivative_i_output * di
      d_c_tilda_input = derivative_c_tilda_output * dg
      self.i_weights_derivative += np.outer(di_input, self.xc)
      self.c_weights_derivative += np.outer(d_c_tilda_input, self.xc)

      dxc = np.dot(self.i_weights.T, di_input)
      dxc += np.dot(self.c_weights.T, d_c_tilda_input)

      self.i_biases_derivative += di_input
      self.c_biases_derivative += d_c_tilda_input

      self.c_weights -= learning_rate * self.c_weights_derivative
      self.i_weights -= learning_rate * self.i_weights_derivative
      self.c_biases -= learning_rate * self.c_biases_derivative
      self.i_biases -= learning_rate * self.i_biases_derivative

      self.c_weights_derivative = np.zeros_like(self.c_weights)
      self.i_weights_derivative = np.zeros_like(self.i_weights)
      self.c_biases_derivative = np.zeros_like(self.c_biases)
      self.i_biases_derivative = np.zeros_like(self.i_biases) 

      return dxc

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