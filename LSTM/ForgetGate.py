import numpy as np

from helpers.useFunctions import sigmoid
class ForgetGate: 
   def __init__(self, parameters, hidden_size, features_number, learning_rate):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.learning_rate = learning_rate
      self.parameters = parameters

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []
      
      self.f_output = []

   def backward(self, derivative_f_output, df):
      df_input = derivative_f_output * df
      self.parameters.increase_f_weights_derivatives(np.outer(df_input, self.xc))
      self.parameters.increase_f_biases_derivatives(df_input)
      dxc = np.dot(self.parameters.get_f_weights().T, df_input)

      return dxc

   def forward(self, x, c_prev = None, h_prev = None):
      if (c_prev is not None and h_prev is not None):
         self.c_prev = c_prev
         self.h_prev = h_prev
      
      self.xc = np.hstack((x, self.h_prev))
      self.f_output = sigmoid(np.dot(self.parameters.get_f_weights(), self.xc) + self.parameters.get_f_biases())
      return self.c_prev, self.f_output
      

   def get_f_output(self):
      return self.f_output

      

