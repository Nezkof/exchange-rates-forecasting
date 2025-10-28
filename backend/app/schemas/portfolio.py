from pydantic import BaseModel
from typing import List

class OptimizationRequest(BaseModel):
   tickers : List[str]
   data_length : int
   control_length : int
   window_size : int
   hidden_size : int
   samples_amount : int
   risk_threshold: int
   capital : int

