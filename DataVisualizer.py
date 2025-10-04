from matplotlib import pyplot as plt
import numpy as np


class DataVisualizer:
   def __init__(self):
      self.data = []
      self.subplots = {}

   def add_subplot(self, label, data):
      if label not in self.subplots:
         self.subplots[label] = []

      for row in data:
         subplot_data = {
            "x_values": row[0],
            "y_values": row[1],
            "color": row[2],
            "marker": row[3],
            "label": row[4]
         }
         self.subplots[label].append(subplot_data)

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


   def build_subplots(self):
      n = len(self.subplots)
      if n == 0:
         return

      fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))
      if n == 1:
         axes = [axes] 

      for ax, (label, plots) in zip(axes, self.subplots.items()):
         for data in plots:
            ax.plot(
               data["x_values"],
               data["y_values"],
               marker=data["marker"],
               color=data["color"],
               label=data["label"]
            )
         ax.set_title(label)
         ax.grid(True)
         ax.legend()

      plt.tight_layout()
      plt.show()