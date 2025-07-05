from matplotlib import pyplot as plt


class DataVisualizer:
   def __init__(self, features_number, train_length, control_length):
      self.features_number = features_number
      self.train_length = train_length
      self.control_length = control_length
   
   def set_train_results(self, data, color):
      self.train_results = data
      self.train_results_color = color

   def set_expected_train_results(self, data, color):
      self.expected_train_results = data
      self.expected_train_results_color = color

   def set_control_results(self, data, color):
      self.control_results = data
      self.control_results_color = color
   
   def set_exprected_control_results(self, data, color):
      self.expected_control_results = data
      self.expected_control_results_color = color
   
   def build_plot(self):      
      plt.figure(figsize=(12, 6))

      train_x = range(self.features_number, self.features_number + self.train_length)
      control_x = range(self.train_length + self.features_number, self.features_number + self.train_length + self.control_length)

      plt.plot(train_x, self.train_results, marker='X', color=self.train_results_color, label='Train Results')
      plt.plot(train_x, self.expected_train_results, marker='o', color=self.expected_train_results_color, label='Expected Train Results')
      plt.plot(control_x, self.control_results, marker='X', color=self.control_results_color, label='Control Results')
      plt.plot(control_x, self.expected_control_results, marker='o', color=self.expected_control_results_color, label='Expected Control Results')

      plt.grid(True)
      plt.legend()
      plt.tight_layout()
      plt.show()
