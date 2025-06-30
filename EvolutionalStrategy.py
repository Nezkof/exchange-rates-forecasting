import random
from helpers.useRandom import generateRandom, generateRandomValues

class EvolutionalStrategy:  
   def __init__(self, population_size, fitness_function, expected_output , bounds, dimensions, mutation_rate=0.1, mutation_strength=0.1):
      self.populationSize = population_size if population_size % 2 == 0 else population_size + 1
      self.lstm = fitness_function
      self.expected_output = expected_output 
      self.bounds = bounds
      self.dimensions = dimensions 
      self.mutation_rate = mutation_rate
      self.mutation_strength = mutation_strength

      self.generatePopulation()
      self.populationErrors = []

   def generateRandomParent(self):
      parent = []
      for j in range(self.dimensions):
         parent.append(generateRandom(self.bounds))
      return parent
   
   def generateChild(self):
      child = []
      
      first_parent = self.getParent()
      second_parent = self.getParent()
      
      for j in range(self.dimensions):
         child.append((first_parent[j] + second_parent[j]) / 2)
      return child
   
   def generatePopulation(self):
      self.population = []

      for i in range(self.populationSize):
         parent = self.generateRandomParent()
         self.population.append(parent)

   def __mean_squared_error(self, predictions, targets):
      return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)

   def calcPopulationErrors(self):
      self.populationErrors = []

      for individ in self.population:
         self.lstm.set_params(individ)
         predictions = self.lstm.fit()  
         error = self.__mean_squared_error(predictions, self.expected_output)
         self.populationErrors.append(error)

   def getParent(self, tournament_size=3):
      indices = generateRandomValues(self.populationSize, tournament_size)

      best_index = indices[0]
      best_error = self.populationErrors[best_index]

      for idx in indices[1:]:
         if self.populationErrors[idx] < best_error:
            best_index = idx
            best_error = self.populationErrors[idx]

      return self.population[best_index]

   def formNewPopulation(self):
      children = []

      for i in range(self.populationSize):
         child = self.generateChild()
         children.append(child)

      self.population.extend(children)

   def mutatePopulation(self):
      for i in range(len(self.population)):
         for j in range(self.dimensions):
            if random.random() < self.mutation_rate:
               mutation = random.uniform(-self.mutation_strength, self.mutation_strength)
               mutated_value = self.population[i][j] + mutation
               
               #  # Обмежуємо в межах bounds
               #  lower_bound, upper_bound = self.bounds
               #  mutated_value = max(lower_bound, min(mutated_value, upper_bound))
                  
               self.population[i][j] = mutated_value

   def selectNextPopulation(self):
      self.population = self.population[:self.populationSize]

   def sortPopulationByError(self):
      combined = list(zip(self.population, self.populationErrors))
      combined.sort(key=lambda pair: pair[1])  

      self.population, self.populationErrors = zip(*combined) 
      self.population = list(self.population)
      self.populationErrors = list(self.populationErrors)

      # sorted_indices = sorted(range(len(self.populationErrors)), key=lambda i: self.populationErrors[i])
      # self.population = [self.population[i] for i in sorted_indices]
      # self.populationErrors = [self.populationErrors[i] for i in sorted_indices]

   def optimize(self, precision):
      self.calcPopulationErrors()
      while(precision < self.populationErrors[0] ):
         self.calcPopulationErrors()
         self.formNewPopulation()
         self.mutatePopulation()
         self.calcPopulationErrors()
         self.sortPopulationByError()
         self.selectNextPopulation()
         print(self.populationErrors[0])
         
      return self.population[0]


      

         
   
