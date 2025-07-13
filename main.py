from datetime import datetime
import math
import time

import numpy as np
from DataVisualizer import DataVisualizer
from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy
from DataProcessor import DataProcessor
from XLSLogger import XLSLogger

from helpers.useFunctions import identity, rastrigin, styblinski_tang, holder_table, parabola

# V0.1
def test_LSTM():
   function = math.sin
   features_number = 5
   data_length = 40
   train_length_coef = 0.8
   train_length = int(data_length * train_length_coef)
   control_length = data_length - train_length
   dataProcessor = DataProcessor(function, features_number, data_length, train_length_coef)
   dataProcessor.form_data_from_file()
   train_sequences, expected_train_results, control_sequences, expected_control_results = dataProcessor.split_data_table()

   hidden_size = 64
   max_epochs = 200
   bounds = [-1, 1] 
   dimensions = ( (((hidden_size + 1) * hidden_size) * 4 + hidden_size) + (hidden_size * 4) + 1)
   population_size = 100
   precision = 0.0001
   parents_per_child = 2
   children_per_parents = 6
   tournament_size = 10
   similarity_coefficient = 0.1
   mutation_rate = 0.7
   mutation_strength = 0.4
   mutation_fade_speed = 0.8
   similarity_coefficient_fade_speed = 0.9
   
   lstm = LSTM(train_sequences, hidden_size)
   evolutionalStrategy = EvolutionalStrategy(
      max_epochs,population_size, lstm, expected_train_results, bounds, dimensions, 
      parents_per_child, children_per_parents, similarity_coefficient, tournament_size, mutation_rate, mutation_strength, mutation_fade_speed, similarity_coefficient_fade_speed
   )

   start = time.time()
   params, minError, minEpoch = evolutionalStrategy.optimize(precision)
   end = time.time()
   time_taken = end - start

   lstm.set_params(params)
   print("LSTM Params:", params)
   train_results = lstm.fit()
   control_results = []

   for sequence in control_sequences:
      control_results.append(lstm.compute(sequence))

   denormalized_train_results = dataProcessor.denormalize(train_results)
   denormalized_train_expected_values = dataProcessor.denormalize(expected_train_results)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_expected_control_results = dataProcessor.denormalize(expected_control_results)
   std_control_error = 0

   for i in range(len(denormalized_control_results)):
      std_control_error += (denormalized_control_results[i] - denormalized_expected_control_results[i])**2
   
   std_control_error = std_control_error / len(denormalized_control_results)

   # for i in range(len(denormalized_control_results)):
      # print(denormalized_control_results[i], "|", denormalized_expected_control_results[i])

   logger = XLSLogger()
   now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   logger.log([
      now, function.__name__, features_number, train_length, control_length, hidden_size,
      minEpoch, minError, time_taken, 
      population_size, parents_per_child, children_per_parents, tournament_size, similarity_coefficient, mutation_rate, mutation_fade_speed, similarity_coefficient_fade_speed,
      std_control_error
   ])

   dataVisualizer = DataVisualizer(features_number, train_length, control_length)
   dataVisualizer.set_train_results(denormalized_train_results, 'blue')
   dataVisualizer.set_expected_train_results(denormalized_train_expected_values, 'red')
   dataVisualizer.set_control_results(denormalized_control_results, 'lightblue')
   dataVisualizer.set_exprected_control_results(denormalized_expected_control_results, 'pink')

   dataVisualizer.build_plot()

def test_evolutional_algorithm():
   fitness_functions = [
      rastrigin,
      styblinski_tang,
      holder_table
   ]

   dimensions = [
      10,
      10,
      2
   ]

   expected_outputs = [
      0, -39.166165 * dimensions[1], -19.2058
   ]

   bounds = [
      [-5.12, 5.12],
      [-5.0, 5.0],
      [-10.0, 10.0]
   ]

   expected_args = [
      [0,0],
      [-2.9035, -2.9035],
      [8.05502, 9.66459]
   ]

   population_size = 1000
   precision = 0.001 
   parents_per_child = 2
   children_per_parents = 7
   tournament_size = 3
   similarity_coefficient = 0.2
   mutation_rate = 0.7
   mutation_strength = 5

   results = []

   evolutionalStrategy = EvolutionalStrategy(
      population_size, 
      fitness_functions[0], 
      expected_outputs[0], 
      bounds[0], 
      dimensions[0], 
      parents_per_child, 
      children_per_parents,
      similarity_coefficient,
      tournament_size,
      mutation_rate, 
      mutation_strength)

   for i in range(len(fitness_functions)):
      evolutionalStrategy.set_fitness_function(fitness_functions[i])
      evolutionalStrategy.set_bounds(bounds[i])
      evolutionalStrategy.set_dimensions(dimensions[i])
      evolutionalStrategy.set_expected_output(expected_outputs[i])
      result = evolutionalStrategy.optimize(precision)
      results.append(result)

   for i in range(len(results)):
      print(fitness_functions[i](results[i]), fitness_functions[i](expected_args[i]))

# V0.2 
def generate_data(function, features_number, data_length, train_length_coef):
   processor = DataProcessor(function, features_number, data_length, train_length_coef)
   X, y = processor.form_data_table()
   return processor.split_data_table()

def test_new_LSTM():
   np.random.seed(0)

# Data sequences settings
   function = math.sin
   data_length = 5000
   train_length_coef = 0.5
   train_length = int(data_length * train_length_coef)
   control_length = data_length - train_length

# LSTM settings
   hidden_size = 256
   output_size = 1
   features_number = 50
   learning_rate = 0.00001
   nodes_amount = 0
   epochs = 1000
   precision = 0.0001

# Data forming
   dataProcessor = DataProcessor(function, features_number, data_length, train_length_coef)
   X, y = dataProcessor.form_data_from_file()
   X_train, y_train, X_control, y_control = dataProcessor.split_data_table()

   lstm = LSTM(hidden_size,features_number, output_size, learning_rate)
   lstm.fit(X_train, y_train, epochs, precision)

   train_results = lstm.compute(X_train)
   control_results = lstm.compute(X_control)

   denormalized_train_results = dataProcessor.denormalize(train_results)
   denormalized_train_expected_values = dataProcessor.denormalize(y_train)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_expected_control_results = dataProcessor.denormalize(y_control)

   denormalized_train_results = dataProcessor.denormalize(train_results)
   denormalized_train_expected_values = dataProcessor.denormalize(y_train)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_expected_control_results = dataProcessor.denormalize(y_control)
   std_control_error = 0

   for i in range(len(denormalized_control_results)):
      std_control_error += (denormalized_control_results[i] - denormalized_expected_control_results[i])**2
   
   std_control_error = std_control_error / len(denormalized_control_results)

   dataVisualizer = DataVisualizer(features_number, train_length, control_length)
   dataVisualizer.set_train_results(denormalized_train_results, 'blue')
   dataVisualizer.set_expected_train_results(denormalized_train_expected_values, 'red')
   dataVisualizer.set_control_results(denormalized_control_results, 'lightblue')
   dataVisualizer.set_exprected_control_results(denormalized_expected_control_results, 'pink')

   dataVisualizer.build_plot()

def main():
   # test_LSTM() 
   # test_evolutional_algorithm()
   test_new_LSTM()

if __name__ == "__main__":
   main()