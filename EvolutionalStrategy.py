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
         expected_output , 
         bounds, 
         dimensions, 
         mutation_rate=0.1, 
         mutation_strength=0.1, 
         max_children_per_parent = 10,
         children_multiplier = 3,
         children_variation = 0.1
   ):
      self.populationSize = population_size if population_size % 2 == 0 else population_size + 1
      self.lstm = fitness_function
      self.expected_output = expected_output 
      self.bounds = bounds
      self.dimensions = dimensions 
      self.mutation_rate = mutation_rate
      self.mutation_strength = mutation_strength
      self.population = []
      self.population_errors = []
      self.max_children_per_parent = max_children_per_parent
      self.children_multiplier = children_multiplier
      self.children_variation = children_variation

      self.membership_function_args = []

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

      for i in range(self.populationSize):
         parent = self.__generate_random_parent()
         self.population.append(parent)

   def __calculate_membership_function_args(self):
      b = get_transition_points_distance(self.expected_output)
      a = 4 * math.log(0.5) / b**2
      self.membership_function_args = [a, self.expected_output]

   # def __mean_squared_error(self, predictions, targets):
   #    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)

   def __calc_population_errors(self):
      self.population_errors = []

      for individ in self.population:
         # self.lstm.set_params(individ)
         # predictions = self.lstm.fit()  
         # error = self.__mean_squared_error(predictions, self.expected_output)
         # self.population_errors.append(error)
         prediction = self.lstm(individ)
         error = abs(self.expected_output - prediction)
         self.population_errors.append(error)

   def __calculate_children_number(self, error):
      children_number = inverse_proportional(self.max_children_per_parent, error)
      children_number = int(round(self.children_multiplier * children_number))
      return max(1, children_number)

   def __generate_parent_children(self, i):
      children = []

      membership_func_belonging_degree = gauss(self.membership_function_args[0], self.membership_function_args[1], self.population_errors[i])

      print(self.population_errors[i], membership_func_belonging_degree)

      # N = self.__calculate_children_number(self.population_errors[i])
      # parent = self.population[i]
      
      # for _ in range(N):
      #    child = []
      #    for x in range(self.dimensions):
      #       mutation = random.gauss(0, self.children_variation)
      #       child_arg = parent[x] + mutation
      #       child.append(child_arg)
      #    children.append(child)

      return children

   def __form_new_population(self):
      children = []
      
      for i in range(len(self.population)):
         parent_children = self.__generate_parent_children(i)
         children.extend(parent_children)

      self.population.extend(children)

   def __mutate_population(self):
      for i in range(len(self.population)):
         if random.random() < self.mutation_rate:
            for j in range(self.dimensions):
               mutation = random.uniform(-self.mutation_strength, self.mutation_strength)
               mutated_value = self.population[i][j] + mutation
               self.population[i][j] = mutated_value

   def __selectNextPopulation(self):
      self.population = self.population[:self.populationSize]

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
      self.__calc_population_errors()

      while(precision < self.population_errors[0]):
         self.__calculate_membership_function_args()
         self.__form_new_population()
         self.__mutate_population()
         self.__calc_population_errors()
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

      

         
   
