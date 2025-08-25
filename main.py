from datetime import datetime
import math
import time
import json

import numpy as np
from DataVisualizer import DataVisualizer
from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy
from DataProcessor import DataProcessor
from ParametersProcessor import ParametersProcessor

# V0.2 
def form_data(dataProcessor, path, column_name):
   dataProcessor.form_data_from_file(path, column_name)
   # dataProcessor.form_data_table()
   return dataProcessor.split_data_table()

def denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results):
   denormalized_train_results = dataProcessor.denormalize(train_results)
   denormalized_train_y = dataProcessor.denormalize(y_train)
   denormalized_control_results = dataProcessor.denormalize(control_results)
   denormalized_control_y = dataProcessor.denormalize(y_control)
   denormalized_pure_control_results = dataProcessor.denormalize(pure_control_results)

   return denormalized_train_results, denormalized_train_y, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results
   
def get_pure_control_results(lstm, x_train, control_length, last_predicted):
    y_out_vector = []
    
    current_sequence = x_train[-1].copy()  
    current_sequence = np.append(current_sequence[1:], last_predicted)
    
    for i in range(control_length):
        print(f"{i * 100 / control_length}%")
        
        y_out = lstm.compute([current_sequence])[-1]
        y_out_vector.append(y_out)
        
        current_sequence = np.append(current_sequence[1:], y_out)
    
    return y_out_vector

def get_control_results(lstm, x_control):
   control_results = []

   for row in x_control:
      y_out = lstm.compute([row])[-1]
      control_results.append(y_out)

   return control_results

def load_file(path):
    with open(path, 'r') as file:
        return json.load(file)

def load_config(config_name):
   config = load_file(f"./configs/{config_name}")
   csv_path = config["csv_path"]
   column_name = config["column_name"]
   hidden_size = config["hidden_size"]
   output_size = config["output_size"]
   features_number = config["features_number"]
   learning_rate = config["learning_rate"]
   learning_rate_decrease_speed = config["learning_rate_decrease_speed"]
   nodes_amount = config["nodes_amount"]
   epochs = config["epochs"]
   precision = config["precision"]
   data_length = config["data_length"]
   control_length = config["control_length"]

   return csv_path, column_name, hidden_size, output_size,features_number, learning_rate, learning_rate_decrease_speed, nodes_amount, epochs, precision, data_length, control_length

def visialize_data(dataVisualizer, train_x, denormalized_train_results, denormalized_train_y, control_x, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results):
   dataVisualizer.add_data(train_x, denormalized_train_results, 'blue', 'X', "Train Results")
   dataVisualizer.add_data(train_x, denormalized_train_y, 'red', 'o', "Expected Train Results")
   dataVisualizer.add_data(control_x, denormalized_control_results, 'lightblue', 'X', "Control Results")
   dataVisualizer.add_data(control_x, denormalized_control_y, 'pink', 'o', "Expected Control Results")
   dataVisualizer.add_data(control_x, denormalized_pure_control_results, 'yellow', 'o', "Pure Control Results")
   dataVisualizer.build_plot()

   print(f"{'Actual value':>15} {'Predicted value':>20} {'Error':>15}")
   print("-" * 50)
   for i in range(len(denormalized_control_y)):
      actual = denormalized_control_y[i]
      predicted = denormalized_pure_control_results[i]
      error = actual - predicted
      print(f"{actual.item():15.6f} {predicted.item():20.6f} {error.item():15.6f}")

def run(config_name, load_weights):
   np.random.seed(0)

# Data loading
   csv_path, column_name, hidden_size, output_size,features_number, learning_rate, learning_rate_decrease_speed, nodes_amount, epochs, precision, data_length, control_length = load_config(config_name)
   train_length = data_length - control_length - features_number

# Data forming
   dataProcessor = DataProcessor(features_number, data_length, train_length, control_length)
   X_train, y_train, X_control, y_control = form_data(dataProcessor, csv_path, column_name)
   train_length = len(X_train)
   control_length = len(X_control)
   data_length = train_length + control_length

   lstm = LSTM(hidden_size,features_number, output_size, learning_rate, learning_rate_decrease_speed)

   parametersProcessor = ParametersProcessor("lstm_weights.npz")
   if (load_weights):
      parametersProcessor.load(lstm.get_parameters())
   else:
      lstm.fit(X_train, y_train, epochs, precision)

   train_results = lstm.compute(X_train, True)
   pure_control_results = get_pure_control_results(lstm, X_train, len(X_control), train_results[-1])
   control_results = lstm.compute(X_control)
   parametersProcessor.save(lstm.get_parameters())

   denormalized_train_results, denormalized_train_y, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results = denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results)
   train_x = range(features_number, features_number + len(denormalized_train_results))
   control_x = range(len(denormalized_train_results) + features_number, features_number + len(denormalized_train_results) + len(denormalized_control_results))

# Data visualizing 
   dataVisualizer = DataVisualizer(features_number, train_length, control_length)
   visialize_data(dataVisualizer, train_x, denormalized_train_results, denormalized_train_y, control_x, denormalized_control_results, denormalized_control_y, denormalized_pure_control_results)


def main():
   config_name = "usd-eur.json"
   load_weights = True
   run(config_name, load_weights)

if __name__ == "__main__":
   main()