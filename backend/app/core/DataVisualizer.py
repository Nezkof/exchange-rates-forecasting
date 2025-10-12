from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import mplcursors


class DataVisualizer:
   def __init__(self):
      self.data = []
      self.subplots = {}

   @staticmethod
   def _show_annotation(sel):
      x_data = sel.artist._x_values_num
      y_data = sel.artist._y_values
      distances = [abs(sel.target[0] - xd) for xd in x_data]
      idx = distances.index(min(distances))
      date_str = sel.artist._x_values_str[idx]
      sel.annotation.set_text(f"x: {date_str}\ny: {y_data[idx].item():.6f}")

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
   
   def build_plot(self):
      plt.figure(figsize=(12,6))

      for data in self.data:
         x_values_dt = [datetime.strptime(d, "%Y-%m-%d") for d in data["x_values"]]
         x_values_num = mdates.date2num(x_values_dt)

         line, = plt.plot(x_values_num, data["y_values"],
                           marker=data["marker"],
                           color=data["color"],
                           label=data["label"])
         line._x_values_str = data["x_values"]
         line._x_values_num = x_values_num
         line._y_values = data["y_values"]

      plt.grid(True)
      plt.legend()
      plt.xticks([])  
      plt.tight_layout()

      cursor = mplcursors.cursor(hover=True)
      cursor.connect("add", self._show_annotation)
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
            x_values_dt = [datetime.strptime(d, "%Y-%m-%d") for d in data["x_values"]]
            x_values_num = mdates.date2num(x_values_dt)

            line, = ax.plot(
               x_values_num,
               data["y_values"],
               marker=data["marker"],
               color=data["color"],
               label=data["label"]
            )

            line._x_values_str = data["x_values"]
            line._x_values_num = x_values_num
            line._y_values = data["y_values"]

         ax.set_title(label)
         ax.grid(True)
         ax.legend()
         ax.set_xticks([])

      cursor = mplcursors.cursor(hover=True)

      cursor.connect("add", self._show_annotation)
      plt.tight_layout()
      plt.show()