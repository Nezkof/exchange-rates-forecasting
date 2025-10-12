import numpy as np

from app.services.helpers.useFunctions import sigmoid, sigmoid_derivative, tanh, tanh_derivative

class InputGate:
   def __init__(self, parameters, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.parameters = parameters

      self.h_prev = np.zeros(hidden_size)
      self.xc = []

      self.s_out = []
      self.i_out = []
   
   def forward(self, xc, h_prev):
      self.h_prev = h_prev

      self.xc = xc
      self.i_out = sigmoid(
         np.dot(self.parameters.get_i_weights(), self.xc) + self.parameters.get_i_biases()
      )
      self.s_out = tanh(
         np.dot(self.parameters.get_s_weights(), self.xc) + self.parameters.get_s_biases()
      )

      return self.i_out, self.s_out
      
   def backward(self, delta_c, i_out, s_out):
      db_i = delta_c * s_out * sigmoid_derivative(i_out)
      dw_i = np.outer(db_i, self.xc)
      db_s = delta_c * i_out * tanh_derivative(s_out)
      dw_s = np.outer(db_s, self.xc)

      self.parameters.increase_i_weights_derivatives(dw_i)
      self.parameters.increase_i_biases_derivatives(db_i)
      self.parameters.increase_s_weights_derivatives(dw_s)
      self.parameters.increase_s_biases_derivatives(db_s)

      delta_h_i = np.dot(self.parameters.get_i_weights().T, db_i)[-self.hidden_size:]
      delta_h_s = np.dot(self.parameters.get_s_weights().T, db_s)[-self.hidden_size:]

      return delta_h_i, delta_h_s

   def get_s_out(self):
      return self.s_out
   
   def get_i_out(self):
      return self.i_out