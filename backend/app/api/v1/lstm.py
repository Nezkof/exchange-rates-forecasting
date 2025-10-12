from fastapi import APIRouter
from app.schemas.lstm import LSTMRequest
from app.services.lstm_service import LSTMService

router = APIRouter()

@router.post("/train")
def train_lstm_endpoint(request: LSTMRequest):
   print(request)
   LSTMService.train_custom_lstm(      
      csv_path=request.csv_path,
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
