import numpy as np
from LSTM.DenseLayer import DenseLayer
from LSTM.ForgetGate import ForgetGate
from LSTM.InputGate import InputGate
from LSTM.LossLayer import LossLayer
from LSTM.OutputGate import OutputGate
from helpers.useFunctions import sigmoid_derivative, tanh, tanh_derivative


class LSTMGate:
   def __init__(self, parameters, hidden_size, features_number):
      self.forget_gate = ForgetGate(parameters, hidden_size, features_number)
      self.input_gate = InputGate(parameters, hidden_size, features_number)
      self.output_gate = OutputGate(parameters, hidden_size, features_number)
      self.dense_layer = DenseLayer(parameters, hidden_size)
      self.loss_layer = LossLayer()

      self.features_number = features_number

      self.c_out = None
      self.h_out = None
      self.y_out = None

      self.loss = 0

   def backward(self, y_train, delta_h_next, delta_c_next):
      o_out = self.output_gate.get_o_out()
      i_out = self.input_gate.get_i_out()
      s_out = self.input_gate.get_s_out()
      f_out = self.forget_gate.get_f_out()

      self.loss = self.loss_layer.calculate_loss(y_train, self.y_out)
      
      loss_gradient = self.loss_layer.calculate_derivative(self.y_out, y_train)
      delta_h = delta_h_next + self.dense_layer.backward(loss_gradient)

      delta_c = delta_c_next + delta_h * o_out * (1 - tanh(self.c_out) ** 2)

      delta_h_o = self.output_gate.backward(delta_h, self.c_out)
      delta_h_i, delta_h_s = self.input_gate.backward(delta_c, i_out, s_out)
      delta_h_f = self.forget_gate.backward(delta_c, f_out, self.c_prev)

      delta_h_total = delta_h_o + delta_h_i + delta_h_s + delta_h_f

      delta_c_prev = delta_c * f_out

      return delta_h_total, delta_c_prev

   def forward(self, row, c_prev, h_prev):
      self.c_prev = c_prev

      xc = np.hstack((row, h_prev))

      f_out = self.forget_gate.forward(xc, c_prev, h_prev)
      i_out, s_out = self.input_gate.forward(xc, h_prev)
      o_out = self.output_gate.forward(xc, h_prev)

      self.c_out = f_out * c_prev + i_out * s_out
      self.h_out = o_out * tanh(self.c_out)

      self.y_out = self.dense_layer.forward(self.h_out)

      return self.c_out, self.h_out, self.y_out

   def calculate_loss(self, y_train):
      self.loss = self.loss_layer.calculate_loss(y_train, self.y_out)
      return self.loss
   


