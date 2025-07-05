from LSTM.ForgetGate import ForgetGate
from LSTM.InputGate import InputGate
from LSTM.OutputGate import OutputGate
from LSTM.DenseGate import DenseGate

class LSTM: 
   def __init_gates(self):
      self.forget_gate = ForgetGate(self.hidden_size)
      self.input_gate = InputGate(self.hidden_size)
      self.output_gate = OutputGate(self.hidden_size)
      self.dense_gate = DenseGate()

   def __set_weights(self, weights): # [forget_w, input_s_w, input_t_w, output_w, dense_w]
      self.forget_gate.set_weights(weights[0])
      self.input_gate.set_sigmoid_weights(weights[1])
      self.input_gate.set_tanh_weights(weights[2])
      self.output_gate.set_weights(weights[3])
      self.dense_gate.set_weights(weights[4])
   
   def __set_biases(self, biases): # [forget_b, input_s_b, input_t_b, output_b, dense_b]
      self.forget_gate.set_biases(biases[0])
      self.input_gate.set_sigmoid_biases(biases[1])
      self.input_gate.set_tanh_biases(biases[2])
      self.output_gate.set_biases(biases[3])
      self.dense_gate.set_biases(biases[4])

   def __struct_params(self, vector, elements_per_subvector):
      gate1 = vector[0 : elements_per_subvector]
      gate2 = vector[elements_per_subvector : 2 * elements_per_subvector]
      gate3 = vector[2 * elements_per_subvector : 3 * elements_per_subvector]
      gate4 = vector[3 * elements_per_subvector : 4 * elements_per_subvector]
      dense = vector[4 * elements_per_subvector : ]  

      return [gate1, gate2, gate3, gate4, dense]

   def __split_params(self, params):
      weights_per_gate = (self.hidden_size + self.features_number) * self.hidden_size
      weights_for_dense_gate = self.hidden_size
      split_index = weights_per_gate * 4 + weights_for_dense_gate

      weights_vector = params[:split_index]
      weights_vector = self.__struct_params(weights_vector, weights_per_gate)

      biases_vector = params[split_index:]
      biases_vector = self.__struct_params(biases_vector, self.hidden_size)

      return [weights_vector, biases_vector]

   def __init__(self, train_sequences, hidden_size):
      self.train_sequences = train_sequences
      self.hidden_size = hidden_size
      self.features_number = 1

      self.c_prev = [0] * hidden_size
      self.h_prev = [0] * hidden_size

      self.__init_gates()

   def set_params(self, params): # [ w1, w2, ... wn, b1, b2, ... bm ]
      params = self.__split_params(params)
      self.__set_weights(params[0])
      self.__set_biases(params[1])

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