from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

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

def main():
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

if __name__ == "__main__":
    main()