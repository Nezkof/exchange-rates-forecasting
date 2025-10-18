from app.core.config import DATASETS_DIR, WEIGHTS_DIR
from typing import List

from app.models.portfolio.markowitz import MarkowitzMethod
from app.utils.portfolio_optimizer import PortfolioOptimizer
from app.services.lstm_service import LSTMService

class PortfolioService:
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
   ):
      daily_returns_path = DATASETS_DIR / f"UAH_History_Data_Returns.csv"

      for ticker in tickers:
         forecast = LSTMService.forecast_custom(csv_type,
                                                ticker, 
                                                data_length,
                                                control_length,
                                                optimizer,
                                                window_size,
                                                hidden_size,
                                             )
         print(forecast.control)
         #TODO

      
      optimizer = MarkowitzMethod(daily_returns_path, tickers)
      optimizer.optimize(samples_amount, len(tickers))
