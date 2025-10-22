import numpy as np

from app.utils.parameters_processor import ParametersProcessor
from app.models.ml.lstm.LSTM import LSTM

class CustomLSTMTrainer:
   def __init__(
         self, 
         optimizer,
         hidden_size, window_size,
         learning_rate, learning_rate_decrease_speed,
         weights_path,
         output_size = 1
      ):
      self.lstm = LSTM (
         optimizer, 
         hidden_size, window_size, output_size,
         learning_rate, learning_rate_decrease_speed 
      )
      self.parameters_processor = ParametersProcessor(weights_path)

   def _calculate_pure_results(self, lstm, x_train, control_length, last_predicted):
      y_out_vector = []
      
      current_sequence = x_train[-1].copy()  
      current_sequence = np.append(current_sequence[1:], last_predicted)
      
      for i in range(control_length):
         y_out = lstm.compute([current_sequence])[-1]
         y_out_vector.append(y_out)
         
         current_sequence = np.append(current_sequence[1:], y_out)
      
      return y_out_vector

   def set_params(self, path):
      self.parameters_processor.set_path(path)
      self.parameters_processor.load(self.lstm.get_parameters())
   
   def fit(self, X_train, Y_train, epochs, precision):
      np.random.seed(0)
      self.lstm.fit(X_train, Y_train, epochs, precision)
      self.parameters_processor.save(self.lstm.get_parameters())

   def compute(self, X_train, X_control):
      train_results = self.lstm.compute(X_train, reset_params=True)
      pure_results = self._calculate_pure_results(self.lstm, X_train, len(X_control), train_results[-1])
      train_results = self.lstm.compute(X_train, reset_params=True)
      control_results = self.lstm.compute(X_control)

      return train_results, pure_results, control_results

   def get_lstm_params(self):
      return self.lstm.get_parameters()