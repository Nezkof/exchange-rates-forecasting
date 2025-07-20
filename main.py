from datetime import datetime
import math
import time
import json

import numpy as np
from DataVisualizer import DataVisualizer
from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy
from DataProcessor import DataProcessor
from XLSLogger import XLSLogger

from helpers.useFunctions import parabola, rastrigin, styblinski_tang, holder_table

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
   denormalized_train_y = dataProcessor.denormalize(expected_train_results)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_control_y = dataProcessor.denormalize(expected_control_results)
   std_control_error = 0

   for i in range(len(denormalized_control_results)):
      std_control_error += (denormalized_control_results[i] - denormalized_control_y[i])**2
   
   std_control_error = std_control_error / len(denormalized_control_results)

   # for i in range(len(denormalized_control_results)):
      # print(denormalized_control_results[i], "|", denormalized_control_y[i])

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
   dataVisualizer.set_expected_train_results(denormalized_train_y, 'red')
   dataVisualizer.set_control_results(denormalized_control_results, 'lightblue')
   dataVisualizer.set_exprected_control_results(denormalized_control_y, 'pink')

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
def form_data(dataProcessor, path, column_name):
   dataProcessor.form_data_from_file(path, column_name)
   # dataProcessor.form_data_table()
   return dataProcessor.split_data_table()

def denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results):
   denormalized_train_results = dataProcessor.denormalize(train_results)
   denormalized_train_y = dataProcessor.denormalize(y_train)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_control_y = dataProcessor.denormalize(y_control)
   denormalized_pure_control_results = dataProcessor.denormalize(pure_control_results)

   return denormalized_train_results, denormalized_train_y, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results
   
def get_pure_control_results(lstm, x_train, control_length, last_predicted):
    y_out_vector = []
    
    current_sequence = x_train[-1].copy()  
    current_sequence = np.append(current_sequence[1:], last_predicted)
    
    for i in range(control_length):
        print(f"{i * 100 / control_length}%")
        
        y_out = lstm.compute([current_sequence])[-1]
        y_out_vector.append(y_out)
        
        current_sequence = np.append(current_sequence[1:], y_out)
    
    return y_out_vector

def get_control_results(lstm, x_control):
   control_results = []

   for row in x_control:
      y_out = lstm.compute([row])[-1]
      control_results.append(y_out)

   return control_results

def load_config(path):
    with open(path, 'r') as file:
        return json.load(file)

def test_new_LSTM():
   np.random.seed(0)

   config = load_config("./configs/usd-uah.json")
   function = parabola
   csv_path = config["csv_path"]
   column_name = config["column_name"]
   hidden_size = config["hidden_size"]
   output_size = config["output_size"]
   features_number = config["features_number"]
   learning_rate = config["learning_rate"]
   learning_rate_decrease_speed = config["learning_rate_decrease_speed"]
   nodes_amount = config["nodes_amount"]
   epochs = config["epochs"]
   precision = config["precision"]
   data_length = config["data_length"]
   control_length = config["control_length"]
   train_length = data_length - control_length - features_number

# Data forming
   dataProcessor = DataProcessor(features_number, data_length, train_length, control_length, function)
   X_train, y_train, X_control, y_control = form_data(dataProcessor, csv_path, column_name)
   train_length = len(X_train)
   control_length = len(X_control)
   data_length = train_length + control_length

   lstm = LSTM(hidden_size,features_number, output_size, learning_rate, learning_rate_decrease_speed)
   lstm.fit(X_train, y_train, epochs, precision)
   train_results = lstm.compute(X_train, True)
   pure_control_results = get_pure_control_results(lstm, X_train, len(X_control), train_results[-1])
   control_results = lstm.compute(X_control)

   denormalized_train_results, denormalized_train_y, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results = denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results)
   train_x = range(features_number, features_number + len(denormalized_train_results))
   control_x = range(len(denormalized_train_results) + features_number, features_number + len(denormalized_train_results) + len(denormalized_control_results))

   dataVisualizer = DataVisualizer(features_number, train_length, control_length)
   dataVisualizer.add_data(train_x, denormalized_train_results, 'blue', 'X', "Train Results")
   dataVisualizer.add_data(train_x, denormalized_train_y, 'red', 'o', "Expected Train Results")
   dataVisualizer.add_data(control_x, denormalized_control_results, 'lightblue', 'X', "Control Results")
   dataVisualizer.add_data(control_x, denormalized_control_y, 'pink', 'o', "Expected Control Results")
   dataVisualizer.add_data(control_x, denormalized_pure_control_results, 'yellow', 'o', "Pure Control Results")
   dataVisualizer.build_plot()

   print(f"{'Actual value':>15} {'Predicted value':>20} {'Error':>15}")
   print("-" * 50)
   for i in range(len(denormalized_control_y)):
      actual = denormalized_control_y[i]
      predicted = denormalized_pure_control_results[i]
      error = actual - predicted
      print(f"{actual:15.6f} {predicted:20.6f} {error:15.6f}")


def main():
   # test_LSTM() 
   # test_evolutional_algorithm()
   test_new_LSTM()

if __name__ == "__main__":
   main()