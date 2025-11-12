from app.core.config import DATASETS_DIR, WEIGHTS_DIR
from typing import List

from app.models.portfolio.markowitz import MarkowitzMethod
# from app.utils.portfolio_optimizer import PortfolioOptimizer
from app.services.lstm_service import LSTMService
from app.utils.csv_processor import CSVHandler

class PortfolioService:
   @staticmethod
   def _process_daily_returns(data_length, history_data_path, daily_returns_path):
      dates, data = CSVHandler.read_csv_all_columns(data_length, history_data_path)
      
      daily_returns_dict_asc = {} 
      
      for col, values in data.items():
         daily_returns_asc = (values[1:] - values[:-1]) / values[:-1]
         
         daily_returns_dict_asc[col] = daily_returns_asc[::-1] 

      return_dates_asc = dates[1:]
      return_dates_desc = return_dates_asc[::-1]

      CSVHandler.write_csv_all_columns(daily_returns_dict_asc, return_dates_desc, daily_returns_path)

   @staticmethod
   def optimize(
      tickers,
      data_length,
      control_length,
      window_size,
      hidden_size,
      samples_amount,
      risk_threshold,
      capital,
      csv_type = 'Data',
      optimizer = 'ADAM'
   ):
      history_data_path = DATASETS_DIR / f"UAH_History_{csv_type}.csv"
      predictions_data_path = DATASETS_DIR / f"UAH_History_{csv_type}_pred.csv"
      daily_returns_path = DATASETS_DIR / f"UAH_History_{csv_type}_pred_returns.csv"
      predictions_results = {}
      risk_threshold = risk_threshold / 100
      rows_num = control_length * 5 + control_length

      for ticker in tickers:
         forecast = LSTMService.forecast_custom(ticker,data_length,control_length, optimizer,window_size, hidden_size)
         predictions_results[ticker] = {
            "pure": forecast.control.pure,
            "control": forecast.control.results,
         }

      CSVHandler.add_to_csv_file(history_data_path, predictions_data_path, predictions_results, "control")
      PortfolioService._process_daily_returns(data_length, predictions_data_path, daily_returns_path)

      optimizer = MarkowitzMethod(daily_returns_path, tickers, risk_threshold, rows_num)
      # optimizer = MarkowitzMethod(DATASETS_DIR / f"UAH_History_Returns_cropped.csv", tickers, risk_threshold, rows_num)
      optimizer.optimize(samples_amount, len(tickers))
      return optimizer.get_chart_data(capital)
