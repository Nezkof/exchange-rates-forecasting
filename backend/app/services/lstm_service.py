from app.core.config import DATASETS_DIR, WEIGHTS_DIR
from app.utils.data_processor import DataProcessor
from app.utils.trainers.custom_lstm_trainer import CustomLSTMTrainer
from app.utils.data_visualizer import DataVisualizer

class LSTMService:
   @staticmethod
   def train_custom_lstm(
      csv_path: str,
      weights_path: str,
      column_name: str,
      data_length: int,
      control_length: int,
      optimizer: str,
      window_size: int,
      hidden_size: int,
      output_size: int,
      learning_rate: float,
      learning_rate_decrease_speed: float,
      epochs: int,
      precision: float,
   ) -> None:
      csv_path = DATASETS_DIR / csv_path
      weights_path = WEIGHTS_DIR / weights_path

      data_processor = DataProcessor(window_size, data_length, control_length)
      data_processor.form_data_from_file(csv_path, column_name)
      X_train, Y_train, X_control, Y_control = data_processor.split_data_table()

      custom_lstm_trainer = CustomLSTMTrainer(
         optimizer,
         hidden_size, window_size, output_size,
         learning_rate, learning_rate_decrease_speed,
         weights_path 
      )
      custom_lstm_trainer.fit(X_train, Y_train, epochs, precision)
      train_results, pure_results, control_results = custom_lstm_trainer.compute(X_train, X_control)
      den_train_y = data_processor.denormalize(Y_train)
      den_control_y = data_processor.denormalize(Y_control)
      den_train_results = data_processor.denormalize(train_results)
      den_control_results = data_processor.denormalize(control_results)
      den_pure_control_results = data_processor.denormalize(pure_results)
      dates = data_processor.get_dates()

      dates_train =  dates[:len(den_train_results)]
      dates_control =  dates[len(den_train_results):]

      data_visualizer = DataVisualizer()
      data_visualizer.add_data(dates_train, den_train_results, 'blue', 'X', "Train Results")
      data_visualizer.add_data(dates_train, den_train_y, 'red', 'o', "Expected Train Results")
      data_visualizer.add_data(dates_control, den_control_results, 'lightblue', 'X', "Control Results")
      data_visualizer.add_data(dates_control, den_control_y, 'pink', 'o', "Expected Control Results")
      data_visualizer.add_data(dates_control, den_pure_control_results, 'yellow', 'o', "Pure Control Results")
      data_visualizer.build_plot()


      # if (load_weights == "True"):
      #    custom_lstm_trainer.set_params(weights_path)
      # else:



