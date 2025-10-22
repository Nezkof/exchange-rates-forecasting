import csv
import numpy as np

class CSVHandler: 
   @staticmethod
   def read_csv_file(data_length, path='', column_name=''):
      values = []
      dates = []

      with open(path, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         headers = next(reader)
         try:
               column_index = headers.index(column_name)
         except ValueError:
               raise ValueError(f"Column '{column_name}' not found in CSV header: {headers}")

         date_index = headers.index("DATE")  

         count = 0
         for row in reader:
               if len(row) > column_index:
                  val = row[column_index]
                  values.append(float(val) if val not in ("", "-") else float('inf'))
                  dates.append(row[date_index])
                  count += 1
                  if count >= data_length:
                     break

      return np.array(dates[::-1]), np.array(values[::-1])
   
   @staticmethod 
   def write_csv_file(values, path='', column_name=''):
      with open(path, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         headers = next(reader)
      
      if column_name not in headers:
         raise ValueError(f"Column '{column_name}' not found in CSV header: {headers}")
      
      column_index = headers.index(column_name)
      
      all_rows = []
      with open(path, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         next(reader) 
         for row in reader:
               all_rows.append(row)
      
      values_to_write = values[::-1]
      
      for i, val in enumerate(values_to_write):
         if i < len(all_rows):
               all_rows[i][column_index] = str(val)
      
      with open(path, mode='w', newline='', encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(headers)
         writer.writerows(all_rows)

   @staticmethod
   def add_to_csv_file(path_source, path_dest, prediction_results, result_type="control"):
      with open(path_source, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         headers = next(reader)
         rows = list(reader)
      
      for ticker in prediction_results:
         col_name = f"{ticker}"
         if col_name not in headers:
               headers.append(col_name)
               for row in rows:
                  row.append("")  
      
      for ticker, results in prediction_results.items():
         col_name = f"{ticker}"
         col_index = headers.index(col_name)
         values = results[result_type]
         
         for i, val in enumerate(values[::-1]):  
               if i < len(rows):
                  rows[i][col_index] = f"{float(val):.6f}"
      
      with open(path_dest, mode='w', newline='', encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(headers)
         writer.writerows(rows)

   @staticmethod
   def write_csv_all_columns(daily_returns_dict, dates, path):
      headers = ["DATE"] + list(daily_returns_dict.keys())
      dates = dates[1:]

      rows = []
      for i, date in enumerate(dates):
         row = [date]
         for col in daily_returns_dict:
            val = daily_returns_dict[col][i]
            row.append(f"{val:.6f}")
         rows.append(row)

      with open(path, mode='w', newline='', encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(headers)
         writer.writerows(rows)

   @staticmethod
   def read_csv_all_columns(length, path):
      data = {}
      dates = []

      with open(path, mode='r', encoding='utf-8') as file:
         reader = csv.reader(file)
         headers = next(reader)
         for h in headers:
               if h != "DATE":
                  data[h] = []

         count = 0
         for row in reader:
               if count >= length:
                  break
               dates.append(row[0])
               for h in headers:
                  if h != "DATE":
                     val = row[headers.index(h)]
                     data[h].append(float(val) if val not in ("", "-") else float('inf'))
               count += 1

      dates = dates[::-1]
      for h in data:
         data[h] = np.array(data[h][::-1])
      return dates, data
