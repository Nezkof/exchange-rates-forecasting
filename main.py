from LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

import math

def standardize_2d(data):
    columns = list(zip(*data))

    standardized_columns = []
    for col in columns:
        mean = sum(col) / len(col)
        std = (sum((x - mean) ** 2 for x in col) / len(col)) ** 0.5
        standardized_columns.append([(x - mean) / std for x in col])

    return [list(row) for row in zip(*standardized_columns)]

def main():
   # x = [[1], [0.5], [0.25], [1]]

   x = [
      [1,1],
      [2,2],
      [3,3],
      [4,4]
   ]

   x_std = standardize_2d(x)

   # print(x_std)

   lstm = LSTM(x_std)

   populationSize = 100
   ff_table_value = 1
   precision = 0.01 
   bounds = [-1000, 1000] 
   dimensions = len(x[0]) * 12

   evolutionalStrategy = EvolutionalStrategy(populationSize, lstm, ff_table_value, bounds, dimensions, 0.4, 0.1)
   weights = evolutionalStrategy.optimize(precision)
   lstm.set_params(weights)
   result = lstm.compute()
   
   print("Result:", result)



if __name__ == "__main__":
    main()
