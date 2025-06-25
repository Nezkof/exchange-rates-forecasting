from helpers.functions import use_tanh, use_sigmoid

class OutputGate:
   def __init__(self, ):
      pass

   def compute(self, x):
      t_temp = use_tanh(self.__c_prev)
      o_temp = use_sigmoid(self.__h_prev * self.__weight[0] + x * self.__weight[1] + self.__bias)
      return [self.__c_prev, o_temp * t_temp]
   
   def set_bias(self, bias):
      self.__bias = bias 
   
   def set_weight(self, weight):
      self.__weight = weight 
   
   def set_c_prev(self, c_prev):
      self.__c_prev = c_prev
   
   def set_h_prev(self, h_prev):
      self.__h_prev = h_prev
