import numpy as np
import csv
from numpy.lib.stride_tricks import sliding_window_view

class DataProcessor:
   def __init__(self, function=None, features_number=5, data_length=1000, train_ratio=0.8):
      self.function = function
      self.features_number = features_number
      self.data_length = data_length
      self.train_length = int(data_length * train_ratio)
      self.train_ratio = train_ratio
      self.control_length = data_length - self.train_length

      self.mean = 0
      self.std = 1

      self.train_data = None  
      self.control_data = None

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
      return np.array(values[::-1]) 

   def __generate_sequences(self, values):
      total_possible = len(values) - self.features_number
      if self.data_length < total_possible:
         values = values[:self.data_length + self.features_number]

      sequences = sliding_window_view(values, self.features_number + 1)
      sequences = sequences[:self.data_length]
      X = sequences[:, :-1]
      y = sequences[:, -1]
      return X, y

   def form_data_from_file(self, path, column_name):
      values = self.__read_data_from_csv(path, column_name)
      X, y = self.__generate_sequences(values)
      X, y = self.__normalize(X, y)
      self.train_length = int(min(self.train_length, self.train_ratio * len(X)))
      self.__split(X, y)
      return X, y

   def form_data_table(self):
      values = np.array([self.function(i) for i in range(self.data_length + self.features_number)])
      X, y = self.__generate_sequences(values)
      X, y = self.__normalize(X, y)
      self.__split(X, y)
      return X, y

   def __split(self, X, y):
      self.train_data = (X[:self.train_length], y[:self.train_length])
      self.control_data = (X[self.train_length:], y[self.train_length:])

   def denormalize(self, normalized_array):
      normalized_array = np.array(normalized_array) 
      return normalized_array * self.std + self.mean

   def split_data_table(self):
      return *self.train_data, *self.control_data

   def get_train_data(self):
      return self.train_data

   def get_control_data(self):
      return self.control_data
