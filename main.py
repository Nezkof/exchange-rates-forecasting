from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

from helpers.useFunctions import rastrigin, styblinski_tang, holder_table

from helpers.useFuzzyLogic import get_transition_points_distance

def denormalize(data, normalized_data):
   mean = sum(data) / len(data)
   std = (sum((x - mean)**2 for x in data) / len(data))**0.5

   return [x * std + mean for x in normalized_data]

def normalize(data):
   mean = sum(data) / len(data)
   std = (sum((x - mean)**2 for x in data) / len(data))**0.5
   return [(x - mean) / std for x in data]

def normalize_2d(data):
   columns = list(zip(*data))
   normalized = []
   for col in columns:
      normalized.append(normalize(col))
   return list(zip(*normalized))

def test_LSTM():
   train_data = [
      [1,2,3],
      [2,3,4],
      [3,4,5],
      [4,5,6],
      [5,6,7],
      [6,7,8],
      [7,8,9],
      [8,9,10],
   ]
   control_data = [3, 4, 5]
   expected_output = [4, 5, 6, 7, 8, 9, 10, 11]
   normalized_train_data = normalize_2d(train_data)
   normalized_expected_output = normalize(expected_output)
   normalized_control_data = normalize(control_data)

   hidden_size = 6
   lstm = LSTM(normalized_train_data, hidden_size)

   population_size = 50
   precision = 0.001 
   bounds = [-5, 5] 
   dimensions = (
      (((hidden_size + 1) * hidden_size) * 4 + hidden_size) + 
      (hidden_size * 4) + 1
   )
   mutation_rate = 0.5
   mutation_strength = 0.2

   evolutionalStrategy = EvolutionalStrategy(
      population_size, 
      lstm, 
      normalized_expected_output, 
      bounds, 
      dimensions, 
      mutation_rate, 
      mutation_strength
   )

   params = evolutionalStrategy.optimize(precision)
   lstm.set_params(params)
   # print(params)
   result = lstm.fit()

   print("Train result:", denormalize(expected_output, result))
   
   result = lstm.compute(normalized_control_data)
   print("Result:", denormalize(control_data, result))

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

   population_size = 100
   precision = 0.001 
   mutation_rate = 0.3
   mutation_strength = 0.9

   results = []

   evolutionalStrategy = EvolutionalStrategy(population_size, fitness_functions[0], expected_outputs[0], bounds[0], dimensions[0], mutation_rate, mutation_strength)

   for i in range(len(fitness_functions)):
      evolutionalStrategy.set_fitness_function(fitness_functions[i])
      evolutionalStrategy.set_bounds(bounds[i])
      evolutionalStrategy.set_dimensions(dimensions[i])
      evolutionalStrategy.set_expected_output(expected_outputs[i])
      result = evolutionalStrategy.optimize(precision)
      results.append(result)

   for i in range(len(results)):
      print(fitness_functions[i](results[i]), fitness_functions[i](expected_args[i]))

def main():
   # test_LSTM() 
   test_evolutional_algorithm()

if __name__ == "__main__":
    main()