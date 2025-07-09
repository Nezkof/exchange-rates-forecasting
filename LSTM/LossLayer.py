import numpy as np


class LossLayer:
   np.random.seed(0)

   def __init__(self):
      pass

   def calculate_loss(self, pred, label):
      return (pred[0] - label) ** 2
   
   def calculate_derivative(self, pred, label):
      diff = np.zeros_like(pred)
      diff[0] = 2 * (pred[0] - label)
      return diff