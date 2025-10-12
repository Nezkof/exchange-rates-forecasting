import random

import numpy as np

def generateRandom(bounds):
   return random.uniform(bounds[0], bounds[1])

def generateRandomValues(populationSize, tournament_size):
   return random.sample(range(populationSize), tournament_size)

def random_array(a, b, *args): 
   np.random.seed(0)
   return np.random.rand(*args) * (b - a) + a