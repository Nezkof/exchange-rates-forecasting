from LSTM.FCL import FCL
from helpers.useFunctions import use_identity

class DenseGate:
   def __init__(self):
      pass
      self.fcl = FCL(use_identity, 1)

   def compute(self, x):
      return self.fcl.calculate(x, self.weights, self.biases)

   def set_biases(self, bias):
      self.biases = bias 
   
   def set_weights(self, weight):
      self.weights = weight   