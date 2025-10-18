import json

def load_file(path):
   with open(path, 'r') as file:
      return json.load(file)
   
def calculate_losses(y, table_y):
   n = len(y)
   errors = [y[i] - table_y[i] for i in range(n)]

   mae = sum(abs(e) for e in errors) / n
   rmse = (sum((e ** 2) for e in errors) / n) ** 0.5
   mape = sum(abs(errors[i] / table_y[i]) for i in range(n)) / n * 100

   return float(mae), float(rmse), float(mape)