import math
import random
from helpers.useRandom import generateRandom, generateRandomValues
from helpers.useFunctions import inverse_proportional, gauss
from helpers.useFuzzyLogic import get_transition_points_distance

class EvolutionalStrategy:  
   def __init__(
         self, 
         population_size, 
         fitness_function, 
         expected_output, 
         bounds, 
         dimensions, 
         parents_per_child,
         children_per_parents,
         similarity_coefficient,
         tournament_size = 3,
         mutation_rate=0.1, 
         mutation_range=0.1, 
   ):
      self.population_size = population_size if population_size % 2 == 0 else population_size + 1
      self.lstm = fitness_function
      self.expected_output = expected_output 
      self.bounds = bounds
      self.dimensions = dimensions 
      self.parents_per_child = parents_per_child
      self.children_per_parents = children_per_parents
      self.similarity_coefficient = similarity_coefficient
      self.tournament_size = tournament_size

      self.mutation_rate = mutation_rate
      self.mutation_range = mutation_range
      
      self.population = []
      self.population_errors = []


   def __generate_random_parent(self):
      parent = []
      for j in range(self.dimensions):
         parent.append(generateRandom(self.bounds))
      return parent
   
   # def __generateChild(self):
      child = []
      
      first_parent = self.getParent()
      second_parent = self.getParent()
      
      for j in range(self.dimensions):
         child.append((first_parent[j] + second_parent[j]) / 2)
      return child
   
   def __generate_population(self):
      self.population = []

      for i in range(self.population_size):
         parent = self.__generate_random_parent()
         self.population.append(parent)

   def __mean_squared_error(self, predictions, targets):
      return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)

   def __calculate_population_errors(self):
      self.population_errors = []

      for individ in self.population:
         self.lstm.set_params(individ)
         predictions = self.lstm.fit()  
         error = self.__mean_squared_error(predictions, self.expected_output)
         self.population_errors.append(error)
         # prediction = self.lstm(individ)
         # error = abs(self.expected_output - prediction)
         # self.population_errors.append(error)
   
   def __tournament_selection(self):
      best_individual = None
      smalles_error = float('inf')

      for _ in range(self.tournament_size):
         random_index = random.randint(0, len(self.population) - 1)
         error = self.population_errors[random_index]

         if error < smalles_error:
               best_individual = self.population[random_index]
               smalles_error = error

      return best_individual

   def __crossover_parents(self, parents):
      child = []

      for i in range(self.dimensions):
         child_gene = 0
         for parent in parents:
            child_gene += parent[i]
         child_gene = child_gene / len(parents)
         child.append(child_gene)

      return child

   def __mutate_individ(self, child):
      mutated = []
      for gene in child:
         noise = random.uniform(-self.similarity_coefficient, self.similarity_coefficient)
         mutated.append(gene + noise)
      return mutated

   def __add_new_population(self):
      children = []

      for _ in range(int(self.population_size / self.parents_per_child)):
         parents = []

         for _ in range(self.parents_per_child):
            parent = self.__tournament_selection()
            parents.append(parent)
         
         for _ in range(self.children_per_parents):
            child = self.__crossover_parents(parents)
            child = self.__mutate_individ(child)
            children.append(child)

      self.population.extend(children)

   def __add_mutated_population(self):
      mutated_population = []

      for i in range(len(self.population)):
         mutated_individ = self.population[i][:]  
         if random.random() < self.mutation_rate:
            for j in range(self.dimensions):
                  noise = random.uniform(-self.mutation_range, self.mutation_range)
                  mutated_individ[j] += noise  
         mutated_population.append(mutated_individ)

      self.population.extend(mutated_population)

   def __selectNextPopulation(self):
      self.population = self.population[:self.population_size]

   def __sort_population_by_error(self):
      combined = list(zip(self.population, self.population_errors))
      combined.sort(key=lambda pair: pair[1])  

      self.population, self.population_errors = zip(*combined) 
      self.population = list(self.population)
      self.population_errors = list(self.population_errors)

      # sorted_indices = sorted(range(len(self.population_errors)), key=lambda i: self.population_errors[i])
      # self.population = [self.population[i] for i in sorted_indices]
      # self.population_errors = [self.population_errors[i] for i in sorted_indices]

   def optimize(self, precision):
      self.__generate_population()
      self.__calculate_population_errors()

      while(precision < self.population_errors[0]):
         self.__add_new_population()
         self.__add_mutated_population()
         self.__calculate_population_errors()
         self.__sort_population_by_error()
         self.__selectNextPopulation()

         print(self.population_errors[0])

      return self.population[0]

   def set_fitness_function(self, fitness_function):
      self.lstm = fitness_function
   
   def set_bounds(self, bounds):
      self.bounds = bounds
   
   def set_dimensions(self, dimensions):
      self.dimensions = dimensions
   
   def set_expected_output(self, expected_output):
      self.expected_output = expected_output

      

         
   
