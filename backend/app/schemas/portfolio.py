from pydantic import BaseModel
from typing import List

class OptimizationRequest(BaseModel):
   csv_type: str
   tickers : List[str]
   data_length : int
   control_length : int
   optimizer : str
   window_size : int
   hidden_size : int
   samples_amount : int
   risk_threshold: int
   capital : int

