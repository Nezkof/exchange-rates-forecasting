import numpy as np
from LSTM.LossLayer import LossLayer
from LSTM.ForgetGate import ForgetGate
from LSTM.InputGate import InputGate
from LSTM.OutputGate import OutputGate

class LSTM: 
   def __init__(self, hidden_size, features_number):      
      self.hidden_size = hidden_size
      self.features_number = features_number

      self.forget_gate = ForgetGate(self.hidden_size, self.features_number)
      self.input_gate = InputGate(self.hidden_size, self.features_number)
      self.output_gate = OutputGate(self.hidden_size, self.features_number)
      self.loss_layer = LossLayer()

   def __backward(self, h_derivative, c_derivative):
      ds = self.output_gate.get_o_output() * h_derivative + c_derivative
      do = self.c_output * h_derivative
      di = self.input_gate.get_c_output() * ds
      dg = self.input_gate.get_i_output() * ds
      df = self.c_prev * ds
      
      print("hello")


   def __calculate_loss(self, cells_output, control_sequence):
      idx = len(control_sequence) - 1

      loss = self.loss_layer.calculate_loss(cells_output[idx], control_sequence[idx])
      h_derivative = self.loss_layer.calculate_derivative(cells_output[idx], control_sequence[idx])
      c_derivative = np.zeros(self.hidden_size)

      self.__backward(h_derivative, c_derivative)


   def forward(self, train_sequences, control_sequence):
      self.c_prev = None
      self.c_output = None
      h_output = None
      cells_output = []
      for row in train_sequences:
         if (self.c_output is not None): 
            self.c_prev = self.c_output
         self.c_output, f_output = self.forget_gate.forward(row, self.c_output, h_output)
         i_output, c_tilda_output = self.input_gate.forward(row, h_output)
         o_output = self.output_gate.forward(row, h_output)

         self.c_output = c_tilda_output * i_output + self.c_output * f_output
         h_output = self.c_output * o_output
         
         cells_output.append(h_output)

      
      loss = self.__calculate_loss(cells_output, control_sequence)



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