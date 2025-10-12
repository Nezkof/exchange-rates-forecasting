import numpy as np

from helpers.useFunctions import sigmoid, sigmoid_derivative, tanh

class OutputGate:
   def __init__(self, parameters, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.parameters = parameters

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []

      self.o_out = []

   def backward(self, delta_h, c_out):
      db = delta_h * tanh(c_out) * sigmoid_derivative(self.o_out)
      dw = np.outer(db, self.xc)

      self.parameters.increase_o_weights_derivatives(dw)
      self.parameters.increase_o_biases_derivatives(db)

      delta_h = np.dot(self.parameters.get_o_weights().T, db)[-self.hidden_size:]

      return delta_h

   def forward(self, xc, h_prev = None):
      self.h_prev = h_prev

      self.xc = xc
      self.o_out = sigmoid(
         np.dot(self.parameters.get_o_weights(), self.xc) + self.parameters.get_o_biases()
      )

      return self.o_out

   def get_o_out(self):
      return self.o_out
   
   def get_c_prev(self):
      return self.c_prev