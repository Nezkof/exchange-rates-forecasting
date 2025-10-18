from app.core.config import DATASETS_DIR, WEIGHTS_DIR
import numpy as np
from typing import List
from app.utils.data_processor import DataProcessor
from app.utils.trainers.custom_lstm_trainer import CustomLSTMTrainer
from app.utils.data_visualizer import DataVisualizer
from app.schemas.lstm import ControlResults, DatasetResults, LSTMTrainingResponse

class LSTMService:
   @staticmethod
   def _flatten_to_float(arr) -> List[float]:
      if isinstance(arr, np.ndarray):
         arr = arr.flatten()
      if hasattr(arr, 'tolist'):
         arr = arr.tolist()
      if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
         arr = [item[0] if isinstance(item, list) else item for item in arr]
      return [float(x) for x in arr]
   
   @staticmethod
   def _flatten_to_string(arr) -> List[str]:
      if isinstance(arr, np.ndarray):
         arr = arr.flatten()
      if hasattr(arr, 'tolist'):
         arr = arr.tolist()
      if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
         arr = [item[0] if isinstance(item, list) else item for item in arr]
      return [str(x) for x in arr]

   @staticmethod
   def train_custom_lstm(
      csv_type: str,
      column_name: str,
      data_length: int,
      control_length: int,
      optimizer: str,
      window_size: int,
      hidden_size: int,
      learning_rate: float,
      learning_rate_decrease_speed: float,
      epochs: int,
      precision: float,
   ) -> None:
      csv_path = DATASETS_DIR / f"UAH_History_{csv_type}.csv"
      weights_path = WEIGHTS_DIR / f"{optimizer}-{column_name}.npz"

      data_processor = DataProcessor(window_size, data_length, control_length)
      data_processor.form_data_from_file(csv_path, column_name)
      X_train, Y_train, X_control, Y_control = data_processor.split_data_table()

      custom_lstm_trainer = CustomLSTMTrainer(
         optimizer,
         hidden_size, window_size,
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

      dates_train = dates[:len(den_train_results)]
      dates_control = dates[len(den_train_results):]

      return LSTMTrainingResponse(
         train=DatasetResults(
            dates=LSTMService._flatten_to_string(dates_train),
            results=LSTMService._flatten_to_float(den_train_results),
            expected=LSTMService._flatten_to_float(den_train_y)
         ),
         control=ControlResults(
            dates=LSTMService._flatten_to_string(dates_control),
            results=LSTMService._flatten_to_float(den_control_results),
            expected=LSTMService._flatten_to_float(den_control_y),
            pure=LSTMService._flatten_to_float(den_pure_control_results)
         )
      )


