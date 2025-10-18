from app.utils.data_visualizer import DataVisualizer

from app.utils.trainers.custom_lstm_trainer import CustomLSTMTrainer
from app.utils.data_processor import DataProcessor
from app.utils.csv_processor import CSVHandler
from app.models.portfolio.markowitz import MarkowitzMethod
from app.utils.helpers import load_file

class PortfolioOptimizer:
   def __init__(
         self,
         tickers,
         samples_amount.
         window_size,
         data_length,
         control_length

         # history_data_path, daily_returns_path, weights_path,

      ):
      self.daily_returns_path = daily_returns_path
      self.tickers = tickers
      self.samples_amount = samples_amount

      self.data_processor = DataProcessor(self.window_size, self.data_length, self.control_length)
      self.dataset = {}
      self.predictions_results = {}

   # def _read_dataset(self):
   #    for ticker in self.tickers:
   #       self.data_processor.form_data_from_file(self.history_data_path, ticker)
   #       X_train, Y_train, X_control, Y_control = self.data_processor.split_data_table()
   #       self.dataset[ticker] = {"train": X_train, "control": X_control}

   # def _write_predicted_dataset(self):
   #    CSVHandler.add_to_csv_file(self.history_data_path, self.predictions_results, "control")

   # def _process_daily_returns(self):
   #    dates, data = CSVHandler.read_csv_all_columns(self.lstm_config['data_length'], self.history_data_path)
   #    daily_returns_dict = {}
   #    for col, values in data.items():
   #       daily_returns_dict[col] = (values[1:] - values[:-1]) / values[:-1]

   #    CSVHandler.write_csv_all_columns(daily_returns_dict, dates, self.daily_returns_path)

   # def predict_rates(self):
   #    self._read_dataset()

   #    custom_lstm_trainer = CustomLSTMTrainer(
   #       self.lstm_config["optimizer"],
   #       self.lstm_config["hidden_size"], self.lstm_config["window_size"], self.lstm_config["output_size"],
   #       self.lstm_config["learning_rate"], self.lstm_config["learning_rate_decrease_speed"],
   #       self.weights_path 
   #    )

   #    for ticker, data in self.dataset.items():
   #       print(ticker)
   #       X_train, X_control = data["train"], data["control"]
   #       optimizer = self.lstm_config["optimizer"]
   #       weights_path = f"{self.weights_path}{optimizer}-{ticker}.npz"
   #       custom_lstm_trainer.set_params(weights_path)

   #       train_results, pure_results, control_results = custom_lstm_trainer.compute(X_train, X_control)

   #       den_train_results = self.data_processor.denormalize(train_results)

   #       self.predictions_results[ticker] = {
   #          "pure": self.data_processor.denormalize(pure_results),
   #          "control": self.data_processor.denormalize(control_results)
   #       }

   #       dates_train = self.data_processor.get_dates()[:len(den_train_results)]
   #       dates_control = self.data_processor.get_dates()[len(den_train_results):]

   #    self._write_predicted_dataset()
   #    self._process_daily_returns()
