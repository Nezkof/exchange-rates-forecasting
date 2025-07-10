import numpy as np
from LSTM.LSTMGate import LSTMGate
from helpers.useFunctions import sigmoid_derivative, tanh_derivative

class LSTM: 
   def __init__(self, hidden_size, features_number, learning_rate, nodes_amount):      
      self.hidden_size = hidden_size
      self.features_number = features_number
      self.learning_rate = learning_rate
      self.nodes = [LSTMGate(hidden_size, features_number) for _ in range(nodes_amount)]

   def __calculate_loss(self, control_sequence):
      idx = len(control_sequence) - 1

      loss = self.nodes[idx].calculate_loss(control_sequence[idx])
      h_derivative = self.nodes[idx].calculate_loss_derivative(control_sequence[idx])
      c_derivative = np.zeros(self.hidden_size)

      self.nodes[idx].backward(h_derivative, c_derivative, self.features_number, self.learning_rate)
      idx -= 1

      while idx >= 0:
         loss += self.nodes[idx].calculate_loss(control_sequence[idx])
         h_derivative = self.nodes[idx].calculate_loss_derivative(control_sequence[idx])
         h_derivative += self.nodes[idx+1].get_bottom_diff_h()
         c_derivative = self.nodes[idx+1].get_bottom_diff_s()
         self.nodes[idx].backward(h_derivative, c_derivative, self.features_number, self.learning_rate)
         idx -= 1 
      
      return loss
   
   def forward(self, train_sequences, control_sequence):
      c_prev = None
      h_prev = None

      for i in range(len(self.nodes)):
         c_prev, h_prev = self.nodes[i].forward(train_sequences[i], c_prev, h_prev)
         
      print("y_pred = [" + ", ".join(["% 2.5f" % self.nodes[idx].get_h_output()[0] for idx in range(len(self.nodes))]) + "]", end=", ")
      
      loss = self.__calculate_loss(control_sequence)
      print("loss:", "%.3e" % loss)


   def fit(self):
      predicted_values = []

      for sequence in self.train_sequences: 
         self.c_prev = [0] * self.hidden_size
         self.h_prev = [0] * self.hidden_size
         
         for x in sequence:
            self.forget_gate.set_c_prev(self.c_prev)
            self.forget_gate.set_h_prev(self.h_prev)
            self.c_prev = self.forget_gate.compute(x)

            self.input_gate.set_c_prev(self.c_prev)
            self.input_gate.set_h_prev(self.h_prev)
            self.c_prev = self.input_gate.compute(x)

            self.output_gate.set_c_prev(self.c_prev)
            self.output_gate.set_h_prev(self.h_prev)
            result = self.output_gate.compute(x)

            self.c_prev = result[0]
            self.h_prev = result[1]

         predicted_value = self.dense_gate.compute(self.h_prev)
         predicted_values.append(predicted_value[0]) 
         
      return predicted_values

   def compute(self, x_sequence):
      for x in x_sequence:
         self.forget_gate.set_c_prev(self.c_prev)
         self.forget_gate.set_h_prev(self.h_prev)
         self.c_prev = self.forget_gate.compute(x)

         self.input_gate.set_c_prev(self.c_prev)
         self.input_gate.set_h_prev(self.h_prev)
         self.c_prev = self.input_gate.compute(x)

         self.output_gate.set_c_prev(self.c_prev)
         self.output_gate.set_h_prev(self.h_prev)
         result = self.output_gate.compute(x)

         self.c_prev = result[0]
         self.h_prev = result[1]

      predicted_value = self.dense_gate.compute(self.h_prev)

      return predicted_value[0]