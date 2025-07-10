from LSTM.ForgetGate import ForgetGate
from LSTM.InputGate import InputGate
from LSTM.LossLayer import LossLayer
from LSTM.OutputGate import OutputGate
from helpers.useFunctions import sigmoid_derivative, tanh_derivative


class LSTMGate:
   def __init__(self, hidden_size, features_number, learning_rate):
      self.forget_gate = ForgetGate(hidden_size, features_number, learning_rate)
      self.input_gate = InputGate(hidden_size, features_number, learning_rate)
      self.output_gate = OutputGate(hidden_size, features_number, learning_rate)
      self.loss_layer = LossLayer()

      self.features_number = features_number

      self.c_output = None
      self.h_output = None

      self.loss = 0
      self.c_prev_grad = 0
      self.h_prev_grad = 0
   
   def forward(self, row, c_prev, h_prev):
      c_prev, f_output = self.forget_gate.forward(row, c_prev, h_prev)
      i_output, c_tilda_output = self.input_gate.forward(row, h_prev)
      o_output = self.output_gate.forward(row, h_prev)

      self.c_output = c_tilda_output * i_output + c_prev * f_output
      self.h_output = self.c_output * o_output
      
      return self.c_output, self.h_output
   
   def backward(self, h_derivative, c_derivative):
      ds = self.output_gate.get_o_output() * h_derivative + c_derivative
      do = self.c_output * h_derivative
      di = self.input_gate.get_c_output() * ds
      dg = self.input_gate.get_i_output() * ds
      df = self.c_output * ds

      dxc = self.forget_gate.backward(sigmoid_derivative(self.forget_gate.get_f_output()), df)
      dxc += self.input_gate.backward(sigmoid_derivative(self.input_gate.get_i_output()), tanh_derivative(self.input_gate.get_c_output()),di, dg)
      dxc += self.output_gate.backward(sigmoid_derivative(self.output_gate.get_o_output()), do)

      self.c_prev_grad = ds * self.input_gate.get_i_output()
      self.h_prev_grad = dxc[self.features_number:]

   def calculate_loss(self, y_out):
      self.loss = self.loss_layer.calculate_loss(self.h_output, y_out)
      return self.loss

   def calculate_loss_derivative(self, y_out):
      self.loss_derivative = self.loss_layer.calculate_derivative(self.h_output, y_out)
      return self.loss_derivative

   def get_h_output(self):
      return self.h_output
   
   def get_c_prev_grad(self):
      return self.c_prev_grad
   
   def get_h_prev_grad(self):
      return self.h_prev_grad