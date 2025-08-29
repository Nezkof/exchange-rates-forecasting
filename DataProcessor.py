import numpy as np
import csv
from numpy.lib.stride_tricks import sliding_window_view

class DataProcessor:
   def __init__(self, features_number, data_length, train_length, control_length, function = None):
      self.function = function
      self.features_number = features_number
      self.data_length = data_length
      self.train_length = train_length
      self.control_length = control_length

      self.mean = 0
      self.std = 1

      self.train_data = None  
      self.control_data = None

      self.last_date = None 

   def __normalize(self, X, y):
      all_values = np.concatenate([X.flatten(), y])
      self.mean = all_values.mean()
      self.std = all_values.std()
      X = (X - self.mean) / self.std
      y = (y - self.mean) / self.std
      return X, y

   def __read_data_from_csv(self, path='', column_name=''):
      values = []
      with open(path, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         headers = next(reader) 
         try:
               column_index = headers.index(column_name)
         except ValueError:
               raise ValueError(f"Column '{column_name}' not found in CSV header: {headers}")

         for row in reader:
               if len(row) > column_index:
                  val = row[column_index]
                  if val in ("", "-"):
                     continue
                  values.append(float(val))
                  self.last_date = row[0]
                  if len(values) >= self.data_length:
                     break  
      return np.array(values[::-1])

   def __generate_sequences(self, values):
      sequences = sliding_window_view(values, self.features_number + 1)
      X = sequences[:, :-1]
      y = sequences[:, -1]
      return X, y

   def __split(self, X, y):
      self.train_data = (X[:self.train_length], y[:self.train_length])
      self.control_data = (X[self.train_length:], y[self.train_length:])

   def form_data_from_file(self, path, column_name):
      if self.data_length < self.features_number + self.train_length + self.control_length:
         raise ValueError(f"data_length is too small for the given features_number and split sizes {self.data_length} < {self.features_number} + {self.train_length} + {self.control_length}")

      values = self.__read_data_from_csv(path, column_name)
      X, y = self.__generate_sequences(values)

      available_length = len(X)
      if self.train_length + self.control_length > available_length:
         raise ValueError(f"Not enough data: need {self.train_length + self.control_length}, available {available_length}")
      
      X, y = self.__normalize(X, y)
      self.__split(X, y)
      return X, y

   def form_data_table(self):
      values = np.array([self.function(i) for i in range(self.data_length)])
      X, y = self.__generate_sequences(values)
      X, y = self.__normalize(X, y)
      self.__split(X, y)
      return X, y

   def denormalize(self, normalized_array):
      normalized_array = np.array(normalized_array) 
      return normalized_array * self.std + self.mean

   def split_data_table(self):
      return *self.train_data, *self.control_data

   def calc_lengths_by_ration(self, data_length, features_number, ratio):
      train_length = int(data_length - features_number * ratio) 
      control_length = data_length - train_length
      return train_length, control_length

   def get_train_data(self):
      return self.train_data

   def get_control_data(self):
      return self.control_data

   def get_last_date(self):
      return self.last_date