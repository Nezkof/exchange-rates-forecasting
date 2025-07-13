import numpy as np
from helpers.useRandom import random_array

class LSTMParameters:
   def __init__(self, hidden_size, features_number, output_size,learning_rate, bounds = [-0.05, 0.05]):
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.hx_length = hidden_size + features_number
      self.learning_rate = learning_rate
      self.bounds = bounds

      self.__init_parameters()

   def __init_parameters(self):
      self.f_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.i_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.s_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.o_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.y_weights = random_array(self.bounds[0], self.bounds[1], self.output_size, self.hidden_size)

      self.f_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.i_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.s_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.o_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.y_biases = random_array(self.bounds[0], self.bounds[1], self.output_size)

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

   def get_s_weights(self):
      return self.s_weights

   def get_i_weights(self):
      return self.i_weights

   def get_f_weights(self):
      return self.f_weights

   def get_o_weights(self):
      return self.o_weights

   def get_y_weights(self):
      return self.y_weights

   def get_s_biases(self):
      return self.s_biases

   def get_i_biases(self):
      return self.i_biases

   def get_f_biases(self):
      return self.f_biases

   def get_o_biases(self):
      return self.o_biases
   
   def get_y_biases(self):
      return self.y_biases

