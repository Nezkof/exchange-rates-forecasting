from matplotlib import pyplot as plt


class DataVisualizer:
   def __init__(self, features_number, train_length, control_length):
      self.features_number = features_number
      self.train_length = train_length
      self.control_length = control_length

      self.data = []
   
   def add_data(self, x_values, y_values, color, marker, label):
      values_data = {
         "x_values": x_values,
         "y_values": y_values,
         "color": color,
         "marker": marker,
         "label": label
      }

      self.data.append(values_data)
   
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

      for data in self.data:
         plt.plot(
            data["x_values"],
            data["y_values"],
            marker=data["marker"],
            color=data["color"],
            label=data["label"]
         )

      plt.grid(True)
      plt.legend()
      plt.tight_layout()
      plt.show()
