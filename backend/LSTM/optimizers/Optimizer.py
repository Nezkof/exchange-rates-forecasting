from abc import ABC, abstractmethod
from helpers.useRandom import random_array

class Optimizer(ABC):
   def __init__(self, hidden_size, features_number, output_size, learning_rate, bounds = [-0.05, 0.05]):
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.hx_length = hidden_size + features_number
      self.learning_rate = learning_rate
      self.bounds = bounds

      self._init_parameters()

   def _init_parameters(self):
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
   
   @abstractmethod
   def update_parameters(self):
      pass

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