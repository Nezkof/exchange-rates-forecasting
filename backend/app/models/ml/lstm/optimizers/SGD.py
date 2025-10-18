import numpy as np

from app.models.ml.lstm.optimizers.Optimizer import Optimizer

class SGD(Optimizer):
   def _init_parameters(self):
      super()._init_parameters()

      self.f_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.i_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.s_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.o_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.y_weights_derivatives = np.zeros((self.output_size,self.hidden_size))

      self.f_biases_derivatives = np.zeros(self.hidden_size)
      self.i_biases_derivatives = np.zeros(self.hidden_size)
      self.s_biases_derivatives = np.zeros(self.hidden_size)
      self.o_biases_derivatives = np.zeros(self.hidden_size)
      self.y_biases_derivatives = np.zeros(self.output_size)

   def update_parameters(self):
      self.f_weights -= self.learning_rate * self.f_weights_derivatives
      self.i_weights -= self.learning_rate * self.i_weights_derivatives
      self.s_weights -= self.learning_rate * self.s_weights_derivatives
      self.o_weights -= self.learning_rate * self.o_weights_derivatives
      self.y_weights -= self.learning_rate * self.y_weights_derivatives
      self.f_biases -= self.learning_rate * self.f_biases_derivatives
      self.i_biases -= self.learning_rate * self.i_biases_derivatives
      self.s_biases -= self.learning_rate * self.s_biases_derivatives
      self.o_biases -= self.learning_rate * self.o_biases_derivatives
      self.y_biases -= self.learning_rate * self.y_biases_derivatives

      self.f_weights_derivatives = np.zeros_like(self.f_weights) 
      self.i_weights_derivatives = np.zeros_like(self.i_weights)
      self.s_weights_derivatives = np.zeros_like(self.s_weights)
      self.o_weights_derivatives = np.zeros_like(self.o_weights) 
      self.y_weights_derivatives = np.zeros((self.output_size ,self.hidden_size))
      self.f_biases_derivatives = np.zeros_like(self.f_biases) 
      self.i_biases_derivatives = np.zeros_like(self.i_biases) 
      self.s_biases_derivatives = np.zeros_like(self.s_biases)
      self.o_biases_derivatives = np.zeros_like(self.o_biases) 
      self.y_biases_derivatives = np.zeros(self.output_size)

   def increase_s_weights_derivatives(self, value):
      self.s_weights_derivatives += value

   def increase_i_weights_derivatives(self, value):
      self.i_weights_derivatives += value

   def increase_f_weights_derivatives(self, value):
      self.f_weights_derivatives += value

   def increase_o_weights_derivatives(self, value):
      self.o_weights_derivatives += value

   def increase_y_weights_derivatives(self, value):
      self.y_weights_derivatives += value

   def increase_s_biases_derivatives(self, value):
      self.s_biases_derivatives += value

   def increase_i_biases_derivatives(self, value):
      self.i_biases_derivatives += value

   def increase_f_biases_derivatives(self, value):
      self.f_biases_derivatives += value

   def increase_o_biases_derivatives(self, value):
      self.o_biases_derivatives += value

   def increase_y_biases_derivatives(self, value):
      self.y_biases_derivatives += value
   