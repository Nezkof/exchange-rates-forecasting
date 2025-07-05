import os
from openpyxl import load_workbook, Workbook

class XLSLogger:
   def __init__(self, filename='statistics.xlsx'):
      self.filename = filename
      if not os.path.exists(self.filename):
         self.workbook = Workbook()
         self.sheet = self.workbook.active
         self.sheet.title = 'Logs'
         self.row_index = 2  
         self.workbook.save(self.filename)
      else:
         self.workbook = load_workbook(self.filename)
         self.sheet = self.workbook.active

         self.row_index = 2
         while self.sheet.cell(row=self.row_index, column=1).value is not None:
            self.row_index += 1

   def log(self, values):
      for col_index, value in enumerate(values, start=1):
         self.sheet.cell(row=self.row_index, column=col_index, value=value)
      self.row_index += 1
      self.workbook.save(self.filename)
      print("Logged successfully")
