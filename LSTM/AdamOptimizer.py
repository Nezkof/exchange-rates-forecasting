import numpy as np
from helpers.useRandom import random_array

class AdamOptimizer:
   def __init__(self, hidden_size, features_number, output_size, learning_rate, bounds = [-0.05, 0.05]):
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.hx_length = hidden_size + features_number
      self.bounds = bounds

      self.learning_rate = learning_rate
      self.beta1 = 0.9
      self.beta2 = 0.999
      self.epsilon = 1e-8
      self.iteration = 0

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

      self.f_weights_m = np.zeros_like(self.f_weights)
      self.f_weights_v = np.zeros_like(self.f_weights)
      self.i_weights_m = np.zeros_like(self.i_weights)
      self.i_weights_v = np.zeros_like(self.i_weights)
      self.s_weights_m = np.zeros_like(self.s_weights)
      self.s_weights_v = np.zeros_like(self.s_weights)
      self.o_weights_m = np.zeros_like(self.o_weights)
      self.o_weights_v = np.zeros_like(self.o_weights)
      self.y_weights_m = np.zeros_like(self.y_weights)
      self.y_weights_v = np.zeros_like(self.y_weights)      
      self.f_biases_m = np.zeros_like(self.f_biases)
      self.f_biases_v = np.zeros_like(self.f_biases)
      self.i_biases_m = np.zeros_like(self.i_biases)
      self.i_biases_v = np.zeros_like(self.i_biases)
      self.s_biases_m = np.zeros_like(self.s_biases)
      self.s_biases_v = np.zeros_like(self.s_biases)
      self.o_biases_m = np.zeros_like(self.o_biases)
      self.o_biases_v = np.zeros_like(self.o_biases)
      self.y_biases_m = np.zeros_like(self.y_biases)
      self.y_biases_v = np.zeros_like(self.y_biases)

   def __adam_update(self, param, grad, m, v):
      m[:] = self.beta1 * m + (1 - self.beta1) * grad
      v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)

      m_hat = m / (1 - self.beta1 ** self.iteration)
      v_hat = v / (1 - self.beta2 ** self.iteration)

      param[:] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

   def __zero_gradients(self):
      self.f_weights_derivatives.fill(0)
      self.i_weights_derivatives.fill(0)
      self.s_weights_derivatives.fill(0)
      self.o_weights_derivatives.fill(0)
      self.y_weights_derivatives.fill(0)

      self.f_biases_derivatives.fill(0)
      self.i_biases_derivatives.fill(0)
      self.s_biases_derivatives.fill(0)
      self.o_biases_derivatives.fill(0)
      self.y_biases_derivatives.fill(0)

   def update_parameters(self):
      self.iteration += 1

      self.__adam_update(self.f_weights, self.f_weights_derivatives, self.f_weights_m, self.f_weights_v)
      self.__adam_update(self.i_weights, self.i_weights_derivatives, self.i_weights_m, self.i_weights_v)
      self.__adam_update(self.s_weights, self.s_weights_derivatives, self.s_weights_m, self.s_weights_v)
      self.__adam_update(self.o_weights, self.o_weights_derivatives, self.o_weights_m, self.o_weights_v)
      self.__adam_update(self.y_weights, self.y_weights_derivatives, self.y_weights_m, self.y_weights_v)

      self.__adam_update(self.f_biases, self.f_biases_derivatives, self.f_biases_m, self.f_biases_v)
      self.__adam_update(self.i_biases, self.i_biases_derivatives, self.i_biases_m, self.i_biases_v)
      self.__adam_update(self.s_biases, self.s_biases_derivatives, self.s_biases_m, self.s_biases_v)
      self.__adam_update(self.o_biases, self.o_biases_derivatives, self.o_biases_m, self.o_biases_v)
      self.__adam_update(self.y_biases, self.y_biases_derivatives, self.y_biases_m, self.y_biases_v)

      self.__zero_gradients()

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

