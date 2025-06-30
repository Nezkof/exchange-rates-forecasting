from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

import math

from LSTM.FCL import FCL
from helpers.useFunctions import use_sigmoid

def main():
   # params = [2.7, 1.63, 2.0, 1.65, 1.41, 0.94, 4.38, -0.19, 1.62, 0.62, -0.32, 0.59]
   train_sequence = [[0], [0.5], [0.25], [1]]

   population_size = 100
   expected_output = [0]
   precision = 0.01 
   bounds = [-100, 100] 
   dimensions = 12
   mutation_rate = 0.5
   mutation_strength = 0.5

   hidden_size = 1
   lstm = LSTM(train_sequence, hidden_size)

   evolutionalStrategy = EvolutionalStrategy(population_size, lstm, expected_output, bounds, dimensions, mutation_rate, mutation_strength)

   params = evolutionalStrategy.optimize(precision)
   lstm.set_params(params)
   result = lstm.fit()

   print("Result:", result)


if __name__ == "__main__":
    main()