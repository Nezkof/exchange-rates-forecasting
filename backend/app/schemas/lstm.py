from pydantic import BaseModel

class LSTMRequest(BaseModel):
   load_weights: str
   csv_path: str
   results_path: str
   weights_path: str
   column_name: str
   data_length: int
   control_length: int
   optimizer: str
   window_size: int
   hidden_size: int
   output_size: int
   learning_rate: float
   learning_rate_decrease_speed: float
   epochs: int
   precision: float
