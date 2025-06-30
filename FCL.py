class FCL:
   def __init__(self, activation_function, hidden_size):
      self.activation_function = activation_function
      self.calculated_vector = [0] * hidden_size

      self.hidden_size = hidden_size
   
   def calculate(self, x_vector, w_vector, b_vector):
      for i in range(self.hidden_size):
         self.calculated_vector[i] = 0 

         for j in range(len(x_vector)):
            self.calculated_vector[i] += x_vector[j] * w_vector[i+j]

         self.calculated_vector[i] = self.activation_function(self.calculated_vector[i] + b_vector[i])

      return self.calculated_vector