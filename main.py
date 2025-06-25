from LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy

import math

def fitnessFunction(args):
   return math.sin(args[0]) * math.cos(args[1]) + args[0]**2 + args[1]**2
   # return (args[0] - 3)**2 + (args[1] + 2)**2
   # return (1 - args[0]) ** 2 + 100 * (args[1] - args[0] ** 2) ** 2

def main():
   x = [0, 0.5, 0.25, 1]

   # lstm = LSTM(x)
   # lstm.compute()

   populationSize = 1000 
   ff_table_value = 0
   bounds = [-50000, 50000] 


   evolutionalStrategy = EvolutionalStrategy(populationSize, fitnessFunction, ff_table_value, bounds, 2, 0.4, 0.1)
   evolutionalStrategy.optimize(0.01)

if __name__ == "__main__":
    main()
