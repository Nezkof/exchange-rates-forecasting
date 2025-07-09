from datetime import datetime
import math
import time

import numpy as np
from DataVisualizer import DataVisualizer
from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy
from DataProcessor import DataProcessor
from XLSLogger import XLSLogger

from helpers.useFunctions import rastrigin, styblinski_tang, holder_table, parabola

def test_LSTM():
   function = math.sin
   features_number = 5
   data_length = 100
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

def test_new_LSTM():
   np.random.seed(0)

# Data sequences settings
   function = np.sin
   data_length = 100
   train_length_coef = 0.8

# LSTM settings
   hidden_size = 100
   features_number = 50 

# Data forming
   # processor = DataProcessor(function, features_number, data_length, train_length_coef)
   # X, y = processor.form_data_table()
   # X_train, y_train, X_control, y_control = processor.split_data_table()

   y_list = [-0.5, 0.2, 0.1, -0.5]
   input_val_arr = [np.array([0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314,
       0.57615733, 0.59204193, 0.57225191, 0.22308163, 0.95274901, 
       0.44712538, 0.84640867, 0.69947928, 0.29743695, 0.81379782, 
       0.39650574, 0.8811032 , 0.58127287, 0.88173536, 0.69253159, 
       0.72525428, 0.50132438, 0.95608363, 0.6439902 , 0.42385505,
       0.60639321, 0.0191932 , 0.30157482, 0.66017354, 0.29007761,
       0.61801543, 0.4287687 , 0.13547406, 0.29828233, 0.56996491,
       0.59087276, 0.57432525, 0.65320082, 0.65210327, 0.43141844,
       0.8965466 , 0.36756187, 0.43586493, 0.89192336, 0.80619399,
       0.70388858, 0.10022689, 0.91948261, 0.7142413 , 0.99884701]), np.array([0.1494483 , 0.86812606, 0.16249293, 0.61555956, 0.12381998,
       0.84800823, 0.80731896, 0.56910074, 0.4071833 , 0.069167  ,
       0.69742877, 0.45354268, 0.7220556 , 0.86638233, 0.97552151,
       0.85580334, 0.01171408, 0.35997806, 0.72999056, 0.17162968,
       0.52103661, 0.05433799, 0.19999652, 0.01852179, 0.7936977 ,
       0.22392469, 0.34535168, 0.92808129, 0.7044144 , 0.03183893,
       0.16469416, 0.6214784 , 0.57722859, 0.23789282, 0.934214  ,
       0.61396596, 0.5356328 , 0.58990998, 0.73012203, 0.311945  ,
       0.39822106, 0.20984375, 0.18619301, 0.94437239, 0.7395508 ,
       0.49045881, 0.22741463, 0.25435648, 0.05802916, 0.43441663]), np.array([0.31179588, 0.69634349, 0.37775184, 0.17960368, 0.02467873,
       0.06724963, 0.67939277, 0.45369684, 0.53657921, 0.89667129,
       0.99033895, 0.21689698, 0.6630782 , 0.26332238, 0.020651  ,
       0.75837865, 0.32001715, 0.38346389, 0.58831711, 0.83104846,
       0.62898184, 0.87265066, 0.27354203, 0.79804683, 0.18563594,
       0.95279166, 0.68748828, 0.21550768, 0.94737059, 0.73085581,
       0.25394164, 0.21331198, 0.51820071, 0.02566272, 0.20747008,
       0.42468547, 0.37416998, 0.46357542, 0.27762871, 0.58678435,
       0.86385561, 0.11753186, 0.51737911, 0.13206811, 0.71685968,
       0.3960597 , 0.56542131, 0.18327984, 0.14484776, 0.48805628]), np.array([0.35561274, 0.94043195, 0.76532525, 0.74866362, 0.90371974,
       0.08342244, 0.55219247, 0.58447607, 0.96193638, 0.29214753,
       0.24082878, 0.10029394, 0.01642963, 0.92952932, 0.66991655,
       0.78515291, 0.28173011, 0.58641017, 0.06395527, 0.4856276 ,
       0.97749514, 0.87650525, 0.33815895, 0.96157015, 0.23170163,
       0.94931882, 0.9413777 , 0.79920259, 0.63044794, 0.87428797,
       0.29302028, 0.84894356, 0.61787669, 0.01323686, 0.34723352,
       0.14814086, 0.98182939, 0.47837031, 0.49739137, 0.63947252,
       0.36858461, 0.13690027, 0.82211773, 0.18984791, 0.51131898,
       0.22431703, 0.09784448, 0.86219152, 0.97291949, 0.96083466])]

   lstm = LSTM(hidden_size,features_number)
   lstm.forward(input_val_arr, y_list)

   # train_results = lstm.fit()


def main():
   # test_LSTM() 
   # test_evolutional_algorithm()
   test_new_LSTM()

if __name__ == "__main__":
   main()