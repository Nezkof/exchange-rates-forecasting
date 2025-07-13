import numpy as np

class LossLayer:
   def calculate_loss(self, y_out, target):
      return 0.5 * np.sum((y_out - target) ** 2)

   def calculate_derivative(self, y_out, target):
      return y_out - target
