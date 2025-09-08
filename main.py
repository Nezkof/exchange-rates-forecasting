from datetime import datetime
import math
import time
import json

import numpy as np
from DataVisualizer import DataVisualizer
# from LSTM.LSTM import LSTM
from EvolutionalStrategy import EvolutionalStrategy
from DataProcessor import DataProcessor
from ParametersProcessor import ParametersProcessor
from XLSLogger import XLSLogger

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

# V0.2 
def form_data(dataProcessor, path, column_name):
   dataProcessor.form_data_from_file(path, column_name)
   return dataProcessor.split_data_table()

def denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results):
   den_train_results = dataProcessor.denormalize(train_results)
   den_train_y = dataProcessor.denormalize(y_train)
   den_control_results = dataProcessor.denormalize(control_results)
   den_control_y = dataProcessor.denormalize(y_control)
   den_pure_control_results = dataProcessor.denormalize(pure_control_results)

   return den_train_results, den_train_y, den_control_results, den_control_y, den_pure_control_results
   
def get_pure_control_results(lstm, x_train, control_length, last_predicted):
    y_out_vector = []
    
    current_sequence = x_train[-1].copy()  
    current_sequence = np.append(current_sequence[1:], last_predicted)
    
    for i in range(control_length):
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

def calculate_losses(y, table_y):
   y = np.array(y)
   table_y = np.array(table_y)

   errors = y - table_y

   mae = np.mean(np.abs(errors))
   rmse = np.sqrt(np.mean(errors ** 2))
   mape = np.mean(np.abs(errors / table_y)) * 100

   return mae, rmse, mape

def load_file(path):
    with open(path, 'r') as file:
        return json.load(file)

def load_config(config_name):
   config = load_file(f"./configs/{config_name}.json")
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
   optimizer = config["optimizer"]

   return csv_path, column_name, hidden_size, output_size,features_number, learning_rate, learning_rate_decrease_speed, nodes_amount, epochs, precision, data_length, control_length, optimizer

def visialize_data(dataVisualizer, train_x, den_train_results, den_train_y, control_x, den_control_results, den_control_y, den_pure_control_results):
   dataVisualizer.add_data(train_x, den_train_results, 'blue', 'X', "Train Results")
   dataVisualizer.add_data(train_x, den_train_y, 'red', 'o', "Expected Train Results")
   dataVisualizer.add_data(control_x, den_control_results, 'lightblue', 'X', "Control Results")
   dataVisualizer.add_data(control_x, den_control_y, 'pink', 'o', "Expected Control Results")
   dataVisualizer.add_data(control_x, den_pure_control_results, 'yellow', 'o', "Pure Control Results")
   dataVisualizer.build_plot()

   print(f"{'Actual value':>15} {'Predicted value':>20} {'Error':>15}")
   print("-" * 50)
   for i in range(len(den_control_y)):
      actual = den_control_y[i]
      predicted = den_pure_control_results[i]
      error = actual - predicted
      print(f"{actual.item():15.6f} {predicted.item():20.6f} {error.item():15.6f}")

def run(config_name, load_weights, weights_file_path, results_file_path):
   np.random.seed(0)

# Data loading
   csv_path, column_name, hidden_size, output_size,features_number, learning_rate, learning_rate_decrease_speed, nodes_amount, epochs, precision, data_length, control_length, optimizer = load_config(config_name)
   train_length = data_length - control_length - features_number

# Data forming
   dataProcessor = DataProcessor(features_number, data_length, train_length, control_length)
   X_train, y_train, X_control, y_control = form_data(dataProcessor, csv_path, column_name)
   train_length = len(X_train)
   control_length = len(X_control)
   data_length = train_length + control_length

   lstm = LSTM(optimizer, hidden_size,features_number, output_size, learning_rate, learning_rate_decrease_speed)

   parametersProcessor = ParametersProcessor(f"./results/weights/{optimizer}-{weights_file_path}")
   if (load_weights):
      parametersProcessor.load(lstm.get_parameters())
   else:
      lstm.fit(X_train, y_train, epochs, precision)

   train_results = lstm.compute(X_train, reset_params=True)
   pure_control_results = get_pure_control_results(lstm, X_train, len(X_control), train_results[-1])
   train_results = lstm.compute(X_train, reset_params=True)
   control_results = lstm.compute(X_control)
   parametersProcessor.save(lstm.get_parameters())

   den_train_results, den_train_y, den_control_results, den_control_y, den_pure_control_results = denormalize_data(dataProcessor, train_results, y_train, control_results, y_control, pure_control_results)
   train_x = range(features_number, features_number + len(den_train_results))
   control_x = range(len(den_train_results) + features_number, features_number + len(den_train_results) + len(den_control_results))

# Data logging
   dataLogger = XLSLogger(f"./results/{results_file_path}")
   MAE, RMSE, MAPE = calculate_losses(den_control_results, den_control_y)
   dataLogger.writeFile(optimizer, MAE, RMSE, MAPE , den_control_y, den_pure_control_results)
   
# Data visualizing 
   dataVisualizer = DataVisualizer(features_number, train_length, control_length)
   visialize_data(dataVisualizer, train_x, den_train_results, den_train_y, control_x, den_control_results, den_control_y, den_pure_control_results)

def run_def_lstm(config_name, load_weights, weights_file_path, results_file_path):
   np.random.seed(0)

# Data loading
   csv_path, column_name, hidden_size, output_size,features_number, learning_rate, learning_rate_decrease_speed, nodes_amount, epochs, precision, data_length, control_length, optimizer = load_config(config_name)
   train_length = data_length - control_length - features_number

# Data forming
   dataProcessor = DataProcessor(features_number, data_length, train_length, control_length)
   X_train, y_train, X_control, y_control = form_data(dataProcessor, csv_path, column_name)

   X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
   X_control = X_control.reshape((X_control.shape[0], X_control.shape[1], 1))

   model = Sequential([
      LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)),
      Dense(1)
   ])

   model.compile(optimizer='adam', loss='mse')
   model.summary()

   model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_control, y_control))

   train_results = model.predict(X_train)
   control_results = model.predict(X_control)

   MAE, RMSE, MAPE = calculate_losses(control_results.flatten(), y_control.flatten())


   print(f"MAE-{MAE}")
   print(f"RMSE-{RMSE}")
   print(f"MAPE-{MAPE}")

   plt.figure(figsize=(12, 6))
   plt.plot(y_control, label="Реальні значення", linewidth=2)
   plt.plot(control_results, label="Прогноз LSTM", linestyle="--", linewidth=2)
   plt.title("Прогнозування на контрольній вибірці")
   plt.xlabel("Часові кроки")
   plt.ylabel("Значення")
   plt.legend()
   plt.grid(True)
   plt.show()

def main():
   config_name = "usd-eur"
   time_stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
   # weights_file_path = f"{config_name}-{time_stamp}.npz"
   # weights_file_path = f"{config_name}-weights.npz"
   weights_file_path = f"1e-4.npz"
   results_file_path = "results.xlsx"
   load_weights = True
   # run(config_name, load_weights, weights_file_path, results_file_path)
   run_def_lstm(config_name, load_weights, weights_file_path, results_file_path)

if __name__ == "__main__":
   main()   