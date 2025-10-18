from pydantic import BaseModel
from typing import List

class DatasetResults(BaseModel):
   dates: List[str] 
   results: List[float]
   expected: List[float]

class ControlResults(BaseModel):
   dates: List[str]
   results: List[float]
   expected: List[float]
   pure: List[float]

class LSTMRequest(BaseModel):
   csv_type: str
   column_name: str
   data_length: int
   control_length: int
   optimizer: str
   window_size: int
   hidden_size: int
   learning_rate: float
   learning_rate_decrease_speed: float
   epochs: int
   precision: float


class LSTMTrainingResponse(BaseModel):
   train: DatasetResults
   control: ControlResults

