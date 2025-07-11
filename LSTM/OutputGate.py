import numpy as np

from helpers.useFunctions import sigmoid, tanh

class OutputGate:
   def __init__(self, parameters, hidden_size, features_number, learning_rate):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.learning_rate = learning_rate
      self.parameters = parameters

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []

   def backward(self, derivative_o_output, do):
      do_input = derivative_o_output * do
      self.parameters.increase_o_weights_derivatives(np.outer(do_input, self.xc))
      self.parameters.increase_o_biases_derivatives(do_input)
      dxc = np.dot(self.parameters.get_o_weights().T, do_input)

      return dxc

   def forward(self, x,  h_prev = None):
      if (h_prev is not None):
         self.h_prev = h_prev

      self.xc = np.hstack((x, self.h_prev))
      self.o_output = sigmoid(np.dot(self.parameters.get_o_weights(), self.xc) + self.parameters.get_o_biases())

      return self.o_output

   def get_o_output(self):
      return self.o_output
   
   def get_c_prev(self):
      return self.c_prev