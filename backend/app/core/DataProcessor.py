import numpy as np
import csv
import math
from numpy.lib.stride_tricks import sliding_window_view

from CSVHandler import CSVHandler

class DataProcessor:
   def __init__(self, window_size, data_length, control_length, function = None):
      self.function = function
      self.window_size = window_size
      self.data_length = data_length
      self.train_length = data_length - control_length - window_size
      self.control_length = control_length

      self.mean = 0
      self.std = 1

      self.train_data = None  
      self.control_data = None
      self.dates = []

   def _normalize(self, X, y):
      all_values = np.concatenate([X.flatten(), y])
      self.mean = all_values.mean()
      self.std = all_values.std()
      X = (X - self.mean) / self.std
      y = (y - self.mean) / self.std
      return X, y
   
   def _fill_empties(self, values):
      for i in range(len(values)):
         if math.isinf(values[i]):
               if i == 0: 
                  j = i + 1
                  while j < len(values) and math.isinf(values[j]):
                     j += 1
                  values[i] = values[j] if j < len(values) else 0.0  
               elif i == len(values) - 1:  
                  j = i - 1
                  while j >= 0 and math.isinf(values[j]):
                     j -= 1
                  values[i] = values[j] if j >= 0 else 0.0  
               else:  
                  left, right = values[i - 1], values[i + 1]
                  li, ri = i - 1, i + 1
                  while li >= 0 and math.isinf(values[li]):
                     li -= 1
                  while ri < len(values) and math.isinf(values[ri]):
                     ri += 1
                  if li >= 0 and ri < len(values):
                     values[i] = (values[li] + values[ri]) / 2
                  elif li >= 0:
                     values[i] = values[li]
                  elif ri < len(values):  
                     values[i] = values[ri]
                  else:
                     values[i] = 0.0  
      return values

   def _generate_sequences(self, values):
      sequences = sliding_window_view(values, self.window_size + 1)
      X = sequences[:, :-1]
      y = sequences[:, -1]
      return X, y

   def _split(self, X, y):
      self.train_data = (X[:self.train_length], y[:self.train_length])
      self.control_data = (X[self.train_length:], y[self.train_length:])

   def denormalize(self, normalized_array):
      normalized_array = np.array(normalized_array) 
      return normalized_array * self.std + self.mean
   
   def form_data_from_file(self, path, column_name):
      if self.data_length < self.window_size + self.train_length + self.control_length:
         raise ValueError(f"data_length is too small for the given window_size and split sizes {self.data_length} < {self.window_size} + {self.train_length} + {self.control_length}")

      self.dates, values = CSVHandler.read_csv_file(self.data_length, path, column_name)

      has_missing = np.isinf(values).any()
      if (has_missing):
         values = self._fill_empties(values)
         CSVHandler.write_csv_file(values, path, column_name)
      X, y = self._generate_sequences(values)

      available_length = len(X)
      if self.train_length + self.control_length > available_length:
         raise ValueError(f"Not enough data: need {self.train_length + self.control_length}, available {available_length}")
      
      X, y = self._normalize(X, y)
      self._split(X, y)
      return X, y
   
   def split_data_table(self):
      return *self.train_data, *self.control_data
   
   def get_dates(self):
      return self.dates[self.window_size:]   
   