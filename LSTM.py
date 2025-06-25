from ForgetGate import ForgetGate
from InputGate import InputGate
from OutputGate import OutputGate

class LSTM: 
   def __init__(self, x):
      self.__c_prev = 0
      self.__h_prev = 0
      self.__x = x

      self.init_gates()

   #TODO_TO_REMOVE
   def init_gates_params(self):
      self.__forget_gate.set_bias(1.62)
      self.__forget_gate.set_weight([2.7, 1.63])

      self.__input_gate.set_bias([0.62, -0.32])
      self.__input_gate.set_weight([
         [2, 1.41],
         [1.65, 0.94]
      ])

      self.__output_gate.set_bias(0.59)
      self.__output_gate.set_weight([4.38, -0.19])

   def init_gates(self):
      self.__forget_gate = ForgetGate()
      self.__input_gate = InputGate()
      self.__output_gate = OutputGate()

      self.init_gates_params()

   def compute(self):
      for x_i in self.__x:
         self.__forget_gate.set_c_prev(self.__c_prev)
         self.__forget_gate.set_h_prev(self.__h_prev)
         self.__c_prev = self.__forget_gate.compute(x_i)

         self.__input_gate.set_c_prev(self.__c_prev)
         self.__input_gate.set_h_prev(self.__h_prev)
         self.__c_prev = self.__input_gate.compute(x_i)

         self.__output_gate.set_c_prev(self.__c_prev)
         self.__output_gate.set_h_prev(self.__h_prev)
         result = self.__output_gate.compute(x_i)

         self.__c_prev = result[0]
         self.__h_prev = result[1]
         
         print(x_i, self.__c_prev, self.__h_prev)   

