import numpy as np

from helpers.useFunctions import sigmoid, tanh

class InputGate:
   def __init__(self, parameters, hidden_size, features_number, learning_rate):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.learning_rate = learning_rate

      self.parameters = parameters

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []
   
   def backward(self, derivative_i_output, derivative_c_tilda_output, di, dg):
      di_input = derivative_i_output * di
      d_c_tilda_input = derivative_c_tilda_output * dg
      self.parameters.increase_i_weights_derivatives(np.outer(di_input, self.xc))
      self.parameters.increase_i_biases_derivatives(di_input)
      self.parameters.increase_c_weights_derivatives(np.outer(d_c_tilda_input, self.xc))
      self.parameters.increase_c_biases_derivatives(d_c_tilda_input)

      dxc = np.dot(self.parameters.get_i_weights().T, di_input)
      dxc += np.dot(self.parameters.get_c_weights().T, d_c_tilda_input)

      return dxc

   def forward(self, x, h_prev = None):
      if (h_prev is not None):
         self.h_prev = h_prev
   
      self.xc = np.hstack((x, self.h_prev))
      self.i_output = sigmoid(np.dot(self.parameters.get_i_weights(), self.xc) + self.parameters.get_i_biases())
      self.c_output = tanh(np.dot(self.parameters.get_c_weights(), self.xc) + self.parameters.get_c_biases())

      return self.i_output, self.c_output

   def get_c_output(self):
      return self.c_output
   
   def get_i_output(self):
      return self.i_output