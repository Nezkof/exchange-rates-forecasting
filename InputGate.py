from helpers.functions import use_sigmoid, use_tanh

class InputGate:
   def __init__(self, ):
      pass
   
   def compute(self, x):
      i_temp = use_sigmoid(
            self.__h_prev * self.__weight[0][0] +
            x * self.__weight[0][1] +
            self.__bias[0]
         )
      
      c_temp = use_tanh(
            self.__h_prev * self.__weight[1][0] +
            x * self.__weight[1][1] +
            self.__bias[1]
      )

      return c_temp * i_temp + self.__c_prev

   def set_bias(self, bias):
      self.__bias = bias # [sigm_bias, tanh_bias]
   
   def set_weight(self, weight):
      self.__weight = weight # [ [sigm_w1, sigm_w2] [tanh_w1, tanh_w2] ]
   
   def set_c_prev(self, c_prev):
      self.__c_prev = c_prev
   
   def set_h_prev(self, h_prev):
      self.__h_prev = h_prev
