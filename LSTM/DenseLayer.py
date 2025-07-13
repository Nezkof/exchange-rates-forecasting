import numpy as np


class DenseLayer: 
   def __init__(self, parameters, hidden_size):
      self.hidden_size = hidden_size
      self.parameters = parameters

      self.h = []
      self.y_out = None

   def backward(self, loss):
      dw = np.outer(loss, self.h)
      db = loss

      self.parameters.increase_y_weights_derivatives(dw)
      self.parameters.increase_y_biases_derivatives(db)

      return np.dot(self.parameters.get_y_weights().T, loss)

   def forward(self, h):
      self.h = h
      self.y_out = np.dot(self.parameters.get_y_weights(), h) + self.parameters.get_y_biases()
      return self.y_out 
   

   