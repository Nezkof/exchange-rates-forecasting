import numpy as np
import csv

class DataProcessor:
   def __init__(self, function, features_number, data_length, train_length):
      self.function = function
      self.features_number = features_number
      self.train_length = int(data_length * train_length)
      self.control_length = data_length - self.train_length
      self.data_length = data_length

      self.train_data = [[], 0]
      self.control_data = [[], 0]

      self.mean = 0
      self.std = 0

   def __normalize_data(self, data):
      X = np.array([x for x, _ in data])
      Y = np.array([y for _, y in data]) 

      all_values = np.concatenate([X.flatten(), Y])
      self.mean = all_values.mean()
      self.std = all_values.std()

      X_standardized = (X - self.mean) / self.std
      y_standardized = (Y - self.mean) / self.std

      data = []

      for i in range(len(X_standardized)):
         row = [X_standardized[i], y_standardized[i]]
         data.append(row)

      return data
   
   def __read_data_from_csv(self):
      values = []
      with open('./data/euro-daily-hist_1999_2022.csv', mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         header = next(reader)  
         
         for row in reader:
            if len(row) >= 2: 
                  value = row[-2]
                  if (value in ("", "-")):
                     continue
                  value = float(value)
                  values.append(value)  

      values = values[::-1]
      
      return values

   def form_data_from_file(self):
      values = self.__read_data_from_csv()
      data_table = []
      index = 0

      for _ in range(min(self.data_length, len(values) - self.features_number)):
         row = []
         sequence = []
         for _ in range(self.features_number):
            sequence.append(values[index])
            index += 1
         row.append(sequence)
         row.append(values[index])
         data_table.append(row)
         index -= self.features_number - 1
      
      data_table = self.__normalize_data(data_table)

      self.train_data = data_table[:self.train_length]
      self.control_data = data_table[self.train_length:]

      return data_table
   
   def form_data_table(self):
      data_table = []
      x = 0

      for _ in range(self.data_length):
         row = []
         sequence = []
         for _ in range(self.features_number):
            sequence.append(self.function(x))
            x += 1
         row.append(sequence)
         row.append(self.function(x))
         data_table.append(row)
         x -= self.features_number - 1

      data_table = self.__normalize_data(data_table)

      self.train_data = data_table[:self.train_length]
      self.control_data = data_table[self.train_length:]

      return data_table
   
   def denormalize(self, normalized_data):
      denormalized = []
      for value in normalized_data:
         denormalized.append(value * self.std + self.mean)

      return denormalized

   def split_data_table(self):
      train_sequences = [x for x, y in self.train_data]
      expected_train_results = [y for x, y in self.train_data]

      control_sequences = [x for x, y in self.control_data]
      expected_control_results = [y for x, y in self.control_data]

      return train_sequences, expected_train_results, control_sequences, expected_control_results

   def get_train_data(self):
      return self.train_data

   def get_control_data(self):
      return self.control_data


   

   