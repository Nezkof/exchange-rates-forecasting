import numpy as np

class ParametersProcessor:
   def __init__(self, filepath="weights.npz"):
      self.filepath = filepath
   
   def save(self, params):
      np.savez(
         self.filepath,
         f_weights=params.f_weights,
         i_weights=params.i_weights,
         s_weights=params.s_weights,
         o_weights=params.o_weights,
         y_weights=params.y_weights,
         f_biases=params.f_biases,
         i_biases=params.i_biases,
         s_biases=params.s_biases,
         o_biases=params.o_biases,
         y_biases=params.y_biases
        )
   
   def load(self, params):
      data = np.load(self.filepath)
      params.f_weights = data["f_weights"]
      params.i_weights = data["i_weights"]
      params.s_weights = data["s_weights"]
      params.o_weights = data["o_weights"]
      params.y_weights = data["y_weights"]

      params.f_biases = data["f_biases"]
      params.i_biases = data["i_biases"]
      params.s_biases = data["s_biases"]
      params.o_biases = data["o_biases"]
      params.y_biases = data["y_biases"]

   def set_path(self, path):
      self.filepath = path