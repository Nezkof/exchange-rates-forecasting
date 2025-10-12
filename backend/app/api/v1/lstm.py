from fastapi import APIRouter
from app.schemas.lstm import LSTMRequest
from app.core.lstm_trainer import run_custom_lstm

router = APIRouter()

@router.post("/train")
def train_lstm_endpoint(request: LSTMRequest):
   run_custom_lstm(
      load_weights=request.load_weights,
      csv_path=request.csv_path,
      results_path=request.results_path,
      weights_path=request.weights_path,
      column_name=request.column_name,
      data_length=request.data_length,
      control_length=request.control_length,
      optimizer=request.optimizer,
      window_size=request.window_size,
      hidden_size=request.hidden_size,
      output_size=request.output_size,
      learning_rate=request.learning_rate,
      learning_rate_decrease_speed=request.learning_rate_decrease_speed,
      epochs=request.epochs,
      precision=request.precision
   )
   return {"status": "success"}
