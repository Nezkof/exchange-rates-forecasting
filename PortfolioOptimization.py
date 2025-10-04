from DataVisualizer import DataVisualizer
from helpers.helpers import load_file

from Markowitz import MarkowitzMethod
from trainers.CustomLSTMTrainer import CustomLSTMTrainer
from DataProcessor import DataProcessor

import pandas as pd

class PortfolioOptimization:
   def __init__(
         self,
         lstm_config_name,
         history_data_path, daily_returns_path, weights_path,
         tickers):
      self.lstm_config = self._load_config(lstm_config_name)
      self.weights_path = weights_path

      self.history_data_path = history_data_path
      self.daily_returns_path = daily_returns_path

      self.tickers = tickers
      self.tickers_amount = len(tickers)
      self.samples_amount = 50000

      self.data_processor = DataProcessor(self.lstm_config["window_size"], self.lstm_config["data_length"], self.lstm_config["control_length"])
      self.data_visualizer = DataVisualizer()
      self.dataset = {}
      self.predictions_results = {}


   def _load_config(self, config_name):
      config = load_file(f"./configs/{config_name}.json")
      return {
         "csv_path": config["csv_path"],
         "weights_path": config["weights_path"],
         "results_path": config["results_path"],
         "column_name": config["column_name"],
         "load_weights": config["load_weights"],
         "hidden_size": config["hidden_size"],
         "output_size": config["output_size"],
         "window_size": config["window_size"],
         "batch_size": config["batch_size"],
         "learning_rate": config["learning_rate"],
         "learning_rate_decrease_speed": config["learning_rate_decrease_speed"],
         "epochs": config["epochs"],
         "precision": config["precision"],
         "data_length": config["data_length"],
         "control_length": config["control_length"],
         "optimizer": config["optimizer"],
   }

   def _read_dataset(self):
      for ticker in self.tickers:
         self.data_processor.form_data_from_file(self.history_data_path, ticker)
         X_train, Y_train, X_control, Y_control = self.data_processor.split_data_table()
         self.dataset[ticker] = {"train": X_train, "control": X_control}

   def _write_predicted_dataset(self):
      df = pd.read_csv(self.history_data_path, parse_dates=["DATE"])
      N = self.lstm_config["control_length"]

      for ticker, preds in self.predictions_results.items():
         df[f"{ticker}_pure"] = df[ticker].copy()
         df[f"{ticker}_control"] = df[ticker].copy()

         df.loc[:N-1, f"{ticker}_pure"] = preds["pure"][:N]
         df.loc[:N-1, f"{ticker}_control"] = preds["control"][:N]

      df.to_csv("./datasets/UAH_History_Data_Updated.csv", index=False)

   def _write_daily_returns(self):
      df = pd.read_csv("./datasets/UAH_History_Data_Updated.csv", parse_dates=["DATE"])
      df = df.sort_values("DATE")
      daily_returns = df.copy()
      for col in ["CNY", "EUR", "USD"]:
         daily_returns[col] = df[col].pct_change()  
      daily_returns = daily_returns.dropna().reset_index(drop=True)
      daily_returns.to_csv(self.daily_returns_path, index=False)

   def predict_rates(self):
      self._read_dataset()

      custom_lstm_trainer = CustomLSTMTrainer(
         self.lstm_config["optimizer"],
         self.lstm_config["hidden_size"], self.lstm_config["window_size"], self.lstm_config["output_size"],
         self.lstm_config["learning_rate"], self.lstm_config["learning_rate_decrease_speed"],
         self.weights_path 
      )

      for ticker, data in self.dataset.items():
         print(ticker)
         X_train, X_control = data["train"], data["control"]
         optimizer = self.lstm_config["optimizer"]
         weights_path = f"{self.weights_path}{optimizer}-{ticker}.npz"
         custom_lstm_trainer.set_params(weights_path)
         train_results, pure_results, control_results = custom_lstm_trainer.compute(X_train, X_control)

         den_train_y = X_train
         den_control_y = X_control
         den_train_results = train_results
         den_control_results = control_results
         den_pure_control_results = pure_results
         
         train_x = range(self.lstm_config['window_size'], self.lstm_config['window_size'] + len(den_train_results))
         control_x = range(len(den_train_results) + self.lstm_config['window_size'], self.lstm_config['window_size'] + len(den_train_results) + len(den_control_results))
         self.data_visualizer.add_data(train_x, den_train_results, 'blue', 'X', "Train Results")
         self.data_visualizer.add_data(train_x, den_train_y, 'red', 'o', "Expected Train Results")
         self.data_visualizer.add_data(control_x, den_control_results, 'lightblue', 'X', "Control Results")
         self.data_visualizer.add_data(control_x, den_control_y, 'pink', 'o', "Expected Control Results")
         self.data_visualizer.add_data(control_x, den_pure_control_results, 'yellow', 'o', "Pure Control Results")
         self.data_visualizer.build_plot()

         # self.predictions_results[ticker] = {
         #    "pure": self.data_processor.denormalize(pure_results),
         #    "control": self.data_processor.denormalize(control_results)
         # }

         # self.data_visualizer.add_subplot(
         #    label=ticker,
         #    data = [
         #       [train_x, den_train_results, 'blue', 'X', "Train Results"],
         #       [train_x, den_train_y, 'red', 'o', "Expected Train Results"],
         #       [control_x, self.predictions_results[ticker]["control"], 'lightblue', 'X', "Control Results"],
         #       [control_x, den_control_y, 'pink', 'o', "Expected Control Results"],
         #       [control_x, self.predictions_results[ticker]['pure'], 'yellow', 'o', "Pure Control Results"]
         #    ]
         # )


      # self.data_visualizer.build_subplots()


      # self._write_predicted_dataset()
      # self._write_daily_returns()

   def calculate_daily_returns(self):
      pass

   def optimize_portfolio(self):
      optimizer = MarkowitzMethod(self.daily_returns_path, self.tickers)
      optimizer.optimize(self.samples_amount, self.tickers_amount)