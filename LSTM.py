from ForgetGate import ForgetGate
from InputGate import InputGate
from OutputGate import OutputGate

class LSTM: 
   def __init__(self, x):
      self.__c_prev = [0, 0]
      self.__h_prev = [0, 0]
      self.x = x
      self.dimensions = len(x[0]) * 12

      self.init_gates()

   def init_gates(self):
      self.__forget_gate = ForgetGate()
      self.__input_gate = InputGate()
      self.__output_gate = OutputGate()

   def set_weights(self, weights):
      self.__forget_gate.set_weight([weights[0], weights[1],weights[2], weights[3]])
      self.__input_gate.set_weight([
         [weights[4], weights[5], weights[6], weights[7]],
         [weights[8], weights[9], weights[10], weights[11]]
      ])
      self.__output_gate.set_weight([weights[12], weights[13], weights[14], weights[15]])
   
   def set_biases(self, biases):
      self.__forget_gate.set_bias([biases[0], biases[1]])
      self.__input_gate.set_bias([ [biases[2], biases[3]], [biases[4], biases[5]]])
      self.__output_gate.set_bias([biases[6], biases[7]])   

   def set_params(self, params):
      split_index = round(self.dimensions * 2 / 3)
      self.set_weights(params[:split_index])
      self.set_biases(params[split_index:])

   def compute(self):
      for x_i in self.x:
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
         
         # print(x_i, self.__c_prev, self.__h_prev)   
      
      # print(self.__h_prev)
      return self.__h_prev[0]

