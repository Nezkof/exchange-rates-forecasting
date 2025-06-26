import random

def generateRandom(bounds):
   return random.uniform(bounds[0], bounds[1])

def generateRandomValues(populationSize, tournament_size):
   return random.sample(range(populationSize), tournament_size)