import numpy as np
from helpers.useRandom import random_array

class LSTMParameters:
   def __init__(self, hidden_size, features_number, learning_rate, bounds = [-0.05, 0.05]):
      self.hidden_size = hidden_size
      self.hx_length = hidden_size + features_number
      self.bounds = bounds
      self.learning_rate = learning_rate

      self.c_weights = []
      self.i_weights = []
      self.f_weights = []
      self.o_weights = []
      
      self.c_biases = []
      self.i_biases = []
      self.f_biases = []
      self.o_biases = []

      self.c_weights_derivatives = []
      self.i_weights_derivatives = []
      self.f_weights_derivatives = []
      self.o_weights_derivatives = []   
   
      self.c_biases_derivatives = []
      self.i_biases_derivatives = []
      self.f_biases_derivatives = []
      self.o_biases_derivatives = []

      self.__init_parameters()


   def __init_parameters(self):
      self.i_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.c_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.f_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.o_weights = random_array(self.bounds[0], self.bounds[1], self.hidden_size, self.hx_length)
      self.i_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.c_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.f_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      self.o_biases = random_array(self.bounds[0], self.bounds[1], self.hidden_size)
      
      self.o_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.f_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.i_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.c_weights_derivatives = np.zeros((self.hidden_size,self.hx_length))
      self.o_biases_derivatives = np.zeros(self.hidden_size)
      self.f_biases_derivatives = np.zeros(self.hidden_size)
      self.c_biases_derivatives = np.zeros(self.hidden_size)
      self.i_biases_derivatives = np.zeros(self.hidden_size)


   def update_parameters(self):
      self.c_weights -= self.learning_rate * self.c_weights_derivatives
      self.i_weights -= self.learning_rate * self.i_weights_derivatives
      self.o_weights -= self.learning_rate * self.o_weights_derivatives
      self.f_weights -= self.learning_rate * self.f_weights_derivatives
      self.f_biases -= self.learning_rate * self.f_biases_derivatives
      self.c_biases -= self.learning_rate * self.c_biases_derivatives
      self.i_biases -= self.learning_rate * self.i_biases_derivatives
      self.o_biases -= self.learning_rate * self.o_biases_derivatives

      self.c_weights_derivatives = np.zeros_like(self.c_weights)
      self.i_weights_derivatives = np.zeros_like(self.i_weights)
      self.o_weights_derivatives = np.zeros_like(self.o_weights) 
      self.f_weights_derivatives = np.zeros_like(self.f_weights) 
      self.c_biases_derivatives = np.zeros_like(self.c_biases)
      self.i_biases_derivatives = np.zeros_like(self.i_biases) 
      self.o_biases_derivatives = np.zeros_like(self.o_biases) 
      self.f_biases_derivatives = np.zeros_like(self.f_biases) 

   def increase_c_weights_derivatives(self, value):
      self.c_weights_derivatives += value

   def increase_i_weights_derivatives(self, value):
      self.i_weights_derivatives += value

   def increase_f_weights_derivatives(self, value):
      self.f_weights_derivatives += value

   def increase_o_weights_derivatives(self, value):
      self.o_weights_derivatives += value

   def increase_c_biases_derivatives(self, value):
      self.c_biases_derivatives += value

   def increase_i_biases_derivatives(self, value):
      self.i_biases_derivatives += value

   def increase_f_biases_derivatives(self, value):
      self.f_biases_derivatives += value

   def increase_o_biases_derivatives(self, value):
      self.o_biases_derivatives += value

   def get_c_weights(self):
      return self.c_weights

   def get_i_weights(self):
      return self.i_weights

   def get_f_weights(self):
      return self.f_weights

   def get_o_weights(self):
      return self.o_weights

   def get_c_biases(self):
      return self.c_biases

   def get_i_biases(self):
      return self.i_biases

   def get_f_biases(self):
      return self.f_biases

   def get_o_biases(self):
      return self.o_biases

