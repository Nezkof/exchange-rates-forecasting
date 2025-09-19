import json

from EvolutionalStrategy import EvolutionalStrategy

from DataVisualizer import DataVisualizer
from DataProcessor import DataProcessor
from Markowitz import MarkowitzMethod
from XLSLogger import XLSLogger

from trainers.CustomLSTMTrainer import CustomLSTMTrainer
from trainers.LibLSTMTrainer import LibLSTMTrainer

# V0.2 
def calculate_losses(y, table_y):
   n = len(y)
   errors = [y[i] - table_y[i] for i in range(n)]

   mae = sum(abs(e) for e in errors) / n
   rmse = (sum((e ** 2) for e in errors) / n) ** 0.5
   mape = sum(abs(errors[i] / table_y[i]) for i in range(n)) / n * 100

   return float(mae), float(rmse), float(mape)

def form_data(dataProcessor, path, column_name):
   dataProcessor.form_data_from_file(path, column_name)
   return dataProcessor.split_data_table()

def load_file(path):
   with open(path, 'r') as file:
      return json.load(file)

def load_config(config_name):
   config = load_file(f"./configs/{config_name}.json")
   csv_path = config["csv_path"]
   weights_path = config["weights_path"]
   column_name = config["column_name"]
   hidden_size = config["hidden_size"]
   output_size = config["output_size"]
   features_number = config["features_number"]
   learning_rate = config["learning_rate"]
   learning_rate_decrease_speed = config["learning_rate_decrease_speed"]
   epochs = config["epochs"]
   precision = config["precision"]
   data_length = config["data_length"]
   control_length = config["control_length"]
   optimizer = config["optimizer"]
   load_weights = config["load_weights"]
   results_path = config["results_path"]
   batch_size = config["batch_size"]

   return csv_path, weights_path, results_path, column_name, load_weights, hidden_size, output_size, features_number, batch_size, learning_rate, learning_rate_decrease_speed, epochs, precision, data_length, control_length, optimizer

def visualize_data(
      window_size, train_length, control_length,
      den_train_results, den_train_y, den_control_results, den_control_y, den_pure_control_results
   ):
   train_x = range(window_size, window_size + len(den_train_results))
   control_x = range(len(den_train_results) + window_size, window_size + len(den_train_results) + len(den_control_results))
   data_visualizer = DataVisualizer(window_size, train_length, control_length)
   data_visualizer.add_data(train_x, den_train_results, 'blue', 'X', "Train Results")
   data_visualizer.add_data(train_x, den_train_y, 'red', 'o', "Expected Train Results")
   data_visualizer.add_data(control_x, den_control_results, 'lightblue', 'X', "Control Results")
   data_visualizer.add_data(control_x, den_control_y, 'pink', 'o', "Expected Control Results")
   data_visualizer.add_data(control_x, den_pure_control_results, 'yellow', 'o', "Pure Control Results")
   data_visualizer.build_plot()

def log_results(path, optimizer, Y_control, control_results, control_pure_results=[]):
   dataLogger = XLSLogger(path)
   MAE, RMSE, MAPE = calculate_losses(control_results, Y_control)
   dataLogger.writeFile(optimizer, MAE, RMSE, MAPE, Y_control, control_results, control_pure_results)

def run_custom_lstm(
      load_weights,
      config_name, csv_path, results_path, weights_path, column_name,
      data_length, control_length, 
      optimizer, 
      window_size, hidden_size, output_size, learning_rate, learning_rate_decrease_speed, epochs, precision
):
   weights_path = f"{weights_path}{optimizer}-{config_name}.npz"
   train_length = data_length - control_length - window_size

   data_processor = DataProcessor(window_size, data_length, train_length, control_length)
   X_train, Y_train, X_control, Y_control = form_data(data_processor, csv_path, column_name)

   custom_lstm_trainer = CustomLSTMTrainer(
      optimizer,
      hidden_size, window_size, output_size,
      learning_rate, learning_rate_decrease_speed,
      weights_path 
   )

   if (load_weights == "True"):
      custom_lstm_trainer.set_params(weights_path)
   else:
      custom_lstm_trainer.fit(X_train, Y_train, epochs, precision)

   train_results, pure_results, control_results = custom_lstm_trainer.compute(X_train, X_control)

   den_train_y = data_processor.denormalize(Y_train)
   den_control_y = data_processor.denormalize(Y_control)
   den_train_results = data_processor.denormalize(train_results)
   den_control_results = data_processor.denormalize(control_results)
   den_pure_control_results = data_processor.denormalize(pure_results)

   log_results(results_path, optimizer, den_control_y, den_control_results, den_pure_control_results)
   visualize_data(
      window_size, train_length, control_length,
      den_train_results, den_train_y, den_control_results, den_control_y, den_pure_control_results
   )

def run_lib_lstm(
      csv_path, results_path, column_name,
      data_length, control_length, 
      window_size, epochs, batch_size, hidden_size, output_size
   ):
   train_length = data_length - control_length - window_size

   data_processor = DataProcessor(window_size, data_length, train_length, control_length)
   X_train, Y_train, X_control, Y_control = form_data(data_processor, csv_path, column_name)

   X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
   X_control = X_control.reshape((X_control.shape[0], X_control.shape[1], 1))

   lib_lstm_trainer = LibLSTMTrainer(hidden_size, output_size, X_train)
   lib_lstm_trainer.fit(
      X_train, Y_train, epochs, batch_size, X_control, Y_control
   )
   control_results = lib_lstm_trainer.compute(X_control)

   den_control_y = data_processor.denormalize(Y_control.flatten())
   den_control_results = data_processor.denormalize(control_results.flatten())

   log_results(results_path, "LibLSTM", den_control_y, den_control_results)

def main():
   csv_path = "./datasets/daily_returns.csv"
   tickers = ['AAPL','JPM','WMT','TGT','MSFT','AMGN']
   samples_amount = 50000 
   tickers_amount = 5

   test = MarkowitzMethod(csv_path, tickers)
   test.optimize(samples_amount, tickers_amount)

   # config_name = "usd-eur"

   # csv_path, weights_path, results_path, column_name, load_weights, hidden_size, output_size, window_size, batch_size, learning_rate, learning_rate_decrease_speed, epochs, precision, data_length, control_length, optimizer = load_config(config_name)
   
   # run_lib_lstm(
   #    csv_path, results_path, column_name,
   #    data_length, control_length, 
   #    window_size, epochs, batch_size, hidden_size, output_size
   # )
   
   # run_custom_lstm(
   #    load_weights,
   #    config_name, csv_path, results_path, weights_path, column_name,
   #    data_length, control_length, 
   #    optimizer, 
   #    window_size, hidden_size, output_size, learning_rate, learning_rate_decrease_speed, epochs, precision
   # )

if __name__ == "__main__":
   main()   