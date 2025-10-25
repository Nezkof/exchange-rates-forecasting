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
      daily_returns_dict = {}
      for col, values in data.items():
         daily_returns_dict[col] = (values[1:] - values[:-1]) / values[:-1]

      CSVHandler.write_csv_all_columns(daily_returns_dict, dates, daily_returns_path)

   @staticmethod
   def optimize(
      csv_type,
      tickers,
      data_length,
      control_length,
      optimizer,
      window_size,
      hidden_size,

      samples_amount,
      risk_threshold,
      capital
   ):
      history_data_path = DATASETS_DIR / f"UAH_History_{csv_type}.csv"
      predictions_data_path = DATASETS_DIR / f"UAH_History_{csv_type}_pred.csv"
      daily_returns_path = DATASETS_DIR / f"UAH_History_{csv_type}_pred_returns.csv"
      predictions_results = {}
      risk_threshold = risk_threshold / 100

      for ticker in tickers:
         forecast = LSTMService.forecast_custom(csv_type,ticker,data_length,control_length, optimizer,window_size, hidden_size)
         predictions_results[ticker] = {
            "pure": forecast.control.pure,
            "control": forecast.control.results,
         }
      
      CSVHandler.add_to_csv_file(history_data_path, predictions_data_path, predictions_results, "control")
      PortfolioService._process_daily_returns(data_length, predictions_data_path, daily_returns_path)

      optimizer = MarkowitzMethod(daily_returns_path, tickers, risk_threshold)
      # optimizer = MarkowitzMethod(DATASETS_DIR / f"UAH_History_Returns.csv", tickers, risk_threshold)
      optimizer.optimize(samples_amount, len(tickers))
      return optimizer.get_chart_data(capital)
