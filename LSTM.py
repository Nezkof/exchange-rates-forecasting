from ForgetGate import ForgetGate
from InputGate import InputGate
from OutputGate import OutputGate

class LSTM: 
   def __init_gates(self):
      self.forget_gate = ForgetGate(self.hidden_size)
      self.input_gate = InputGate(self.hidden_size)
      self.output_gate = OutputGate(self.hidden_size)

   def __set_weights(self, weights): # [forget_w, input_s_w, input_t_w, output_w]
      self.forget_gate.set_weights(weights[0])
      self.input_gate.set_sigmoid_weights(weights[1])
      self.input_gate.set_tanh_weights(weights[2])
      self.output_gate.set_weight(weights[3])
   
   def __set_biases(self, biases): # [forget_b, input_s_b, input_t_b, output_b]
      self.forget_gate.set_biases(biases[0])
      self.input_gate.set_sigmoid_biases(biases[1])
      self.input_gate.set_tanh_biases(biases[2])
      self.output_gate.set_biases(biases[3])

   def __split_params(self, params):
      weights_per_gate = self.hidden_size + self.features_number
      split_index = weights_per_gate * 4

      weights_vector = params[:split_index]
      weights_vector = [weights_vector[i * weights_per_gate : (i + 1) * weights_per_gate] for i in range(4)]

      biases_vector = params[split_index:]
      biases_vector = [biases_vector[i * self.hidden_size : (i + 1) * self.hidden_size] for i in range(4)]

      return [weights_vector, biases_vector]

   def __init__(self, train_sequences, hidden_size):
      self.train_sequences = train_sequences
      self.hidden_size = hidden_size
      self.features_number = len(train_sequences[0])

      self.c_prev = [0] * hidden_size
      self.h_prev = [0] * hidden_size

      self.__init_gates()

   def set_params(self, params): # [ w1, w2, ... wn, b1, b2, ... bm ]
      params = self.__split_params(params)
      self.__set_weights(params[0])
      self.__set_biases(params[1])

   def fit(self):
      sequence = self.train_sequences

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
      
      return(self.h_prev)


      # results = []
      # for sequence in self.train_sequences:
      #    self.c_prev = [0] * self.hidden_size
      #    self.h_prev = [0] * self.hidden_size
      #    for x_i in sequence:
      #       self.forget_gate.set_c_prev(self.c_prev)
      #       self.forget_gate.set_h_prev(self.h_prev)
      #       self.c_prev = self.forget_gate.compute(x_i)

      #       self.input_gate.set_c_prev(self.c_prev)
      #       self.input_gate.set_h_prev(self.h_prev)
      #       self.c_prev = self.input_gate.compute(x_i)

      #       self.output_gate.set_c_prev(self.c_prev)
      #       self.output_gate.set_h_prev(self.h_prev)
      #       result = self.output_gate.compute(x_i)

      #       self.c_prev = result[0]
      #       self.h_prev = result[1]
      #    results.append(self.h_prev[0])  
      # return results

