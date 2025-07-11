import os
import numpy as np
from LSTM.LSTMGate import LSTMGate
from LSTM.LSTMParameters import LSTMParameters
from helpers.useFunctions import sigmoid_derivative, tanh_derivative

class LSTM: 
   def __init__(self, hidden_size, features_number, learning_rate, nodes_amount):      
      self.hidden_size = hidden_size
      self.features_number = features_number
      self.lstm_parameters = LSTMParameters(hidden_size, features_number, learning_rate)
      self.nodes = [LSTMGate(self.lstm_parameters, self.hidden_size, self.features_number, learning_rate) for _ in range(nodes_amount)]

   def __update_coefficients(self):
      self.lstm_parameters.update_parameters()

   def __calculate_loss(self, control_sequence):
      idx = len(control_sequence) - 1

      loss = self.nodes[idx].calculate_loss(control_sequence[idx])
      h_derivative = self.nodes[idx].calculate_loss_derivative(control_sequence[idx])
      c_derivative = np.zeros(self.hidden_size)

      self.nodes[idx].backward(h_derivative, c_derivative)
      idx -= 1

      while idx >= 0:
         loss += self.nodes[idx].calculate_loss(control_sequence[idx])
         h_derivative = self.nodes[idx].calculate_loss_derivative(control_sequence[idx])
         h_derivative += self.nodes[idx+1].get_h_prev_grad()
         c_derivative = self.nodes[idx+1].get_c_prev_grad()
         self.nodes[idx].backward(h_derivative, c_derivative)
         idx -= 1 
      
      return loss
   
   def __forward(self, train_sequences):
      c_prev = None
      h_prev = None

      for i in range(len(self.nodes)):
         c_prev, h_prev = self.nodes[i].forward(train_sequences[i], c_prev, h_prev)

      return [node.get_h_output()[0] for node in self.nodes]

   def fit(self, train_sequences, control_sequence, epochs, precision):
      epoch = 0
      loss = np.inf
      while epoch < epochs and loss > precision:
         output = self.__forward(train_sequences)
         loss = self.__calculate_loss(control_sequence)
         self.__update_coefficients()
         
         clear = lambda: os.system('cls')
         clear()
         formatted = ", ".join(f"{val:.5f}" for val in output)
         print(f"Epoch: {epoch} | output: [{formatted}]")
         print("loss:", "%.3e" % loss)
         epoch += 1
   
   def compute(self, sequence):
      return self.__forward(sequence)