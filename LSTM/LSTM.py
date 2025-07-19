import os
import numpy as np
from LSTM.LSTMGate import LSTMGate
from LSTM.LSTMParameters import LSTMParameters
from helpers.useFunctions import sigmoid_derivative, tanh_derivative

class LSTM: 
   def __init__(self, hidden_size, features_number, output_size, learning_rate, lr_decrease_speed):      
      self.hidden_size = hidden_size
      self.features_number = features_number
      self.learning_rate = learning_rate
      self.lr_decrease_speed = lr_decrease_speed
      self.lstm_parameters = LSTMParameters(hidden_size, features_number, output_size, learning_rate)
      self.nodes = []

      self.c_prev = np.zeros(self.hidden_size)
      self.h_prev = np.zeros(self.hidden_size)

   def __forward(self, train_sequences):
      lstm_out = []

      for i in range(len(train_sequences)):
         self.c_prev, self.h_prev, y_out = self.nodes[i].forward(train_sequences[i], self.c_prev, self.h_prev)
         lstm_out.append(y_out)

      return lstm_out
   
   def __backward(self, y_train):
      delta_h_next = np.zeros(self.hidden_size)
      delta_c_next = np.zeros(self.hidden_size)
      avg_loss = 0

      for i in reversed(range(len(y_train))):
         delta_h_next, delta_c_next = self.nodes[i].backward(
               y_train[i], delta_h_next, delta_c_next
         )
         avg_loss += self.nodes[i].loss
      
      avg_loss = avg_loss / len(self.nodes)

      return avg_loss
   
   def __update_coefficients(self):
      self.lstm_parameters.update_parameters()

   def fit(self, x_train, y_train, epochs, precision):
      self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number) for _ in range(len(x_train))]

      epoch = 0
      loss = np.inf
      while epoch < epochs and loss > precision:
         self.c_prev = np.zeros(self.hidden_size)
         self.h_prev = np.zeros(self.hidden_size)
         lstm_out = self.__forward(x_train)
         loss = self.__backward(y_train)
         self.__update_coefficients()

         ## LOGGER FUNCTIONALITY
         clear = lambda: os.system('cls')
         clear()
         # formatted = ", ".join(f"{val:.5f}" for val in lstm_out)
         print(f"Epoch: {epoch} ")
         # print(f"output: [{formatted}]")
         print("loss:", "%.3e" % loss)
         print("lr:", "%.3e" % self.learning_rate)
         ## 

         self.learning_rate *= self.lr_decrease_speed

         epoch += 1

   def compute(self, sequence, reset_params = False):
      if (reset_params == True):
         self.c_prev = np.zeros(self.hidden_size)
         self.h_prev = np.zeros(self.hidden_size)
      self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number) for _ in range(len(sequence))]
      lstm_out = self.__forward(sequence)
      return lstm_out