import numpy as np
from app.services.helpers.useFunctions import sigmoid, sigmoid_derivative, tanh

class ForgetGate: 
   def __init__(self, parameters, hidden_size, features_number):
      np.random.seed(0)
      
      self.hidden_size = hidden_size
      self.parameters = parameters

      self.c_prev = np.zeros(hidden_size)
      self.h_prev = np.zeros(hidden_size)
      self.xc = []
      
      self.f_out = []

   def forward(self, xc, c_prev, h_prev):
      self.c_prev = c_prev
      self.h_prev = h_prev

      self.xc = xc
      self.f_out = sigmoid(
         np.dot(self.parameters.get_f_weights(), self.xc) + self.parameters.get_f_biases()
      )

      return self.f_out
   
   def backward(self, delta_c, f_out, c_prev):
      db = delta_c * c_prev * sigmoid_derivative(f_out)
      dw = np.outer(db, self.xc)

      self.parameters.increase_f_weights_derivatives(dw)
      self.parameters.increase_f_biases_derivatives(db)

      delta_h_f = np.dot(self.parameters.get_f_weights().T, db)[-self.hidden_size:]
      
      return delta_h_f
      
   def get_f_out(self):
      return self.f_out

      

