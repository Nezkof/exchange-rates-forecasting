from LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

import math

from FCL import FCL
from helpers.useFunctions import use_sigmoid

def main():
   # populationSize = 100
   # ff_table_value = 1
   # precision = 0.01 
   # bounds = [-1000, 1000] 
   # dimensions = len(x[0]) * 12

   # evolutionalStrategy = EvolutionalStrategy(populationSize, lstm, ff_table_value, bounds, dimensions, 0.4, 0.1)
   # weights = evolutionalStrategy.optimize(precision)
   # lstm.set_params(weights)
   # result = lstm.compute()

   train_sequence = [[0], [0.5], [0.25], [1]]
   params = [2.7, 1.63, 2.0, 1.65, 1.41, 0.94, 4.38, -0.19, 1.62, 0.62, -0.32, 0.59]


   hidden_size = 1
   lstm = LSTM(train_sequence, hidden_size)
   lstm.set_params(params)
   result = lstm.fit()

   print("Result:", result)


if __name__ == "__main__":
    main()