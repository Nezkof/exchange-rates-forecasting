from fastapi import APIRouter
from app.schemas.lstm import LSTMTrainRequest, LSTMForecastRequest
from app.services.lstm_service import LSTMService

router = APIRouter()

@router.post("/train")
def train_lstm_endpoint(request: LSTMTrainRequest):
   response = LSTMService.train_custom(      
      csv_type=request.csv_type,
      column_name=request.column_name,
      data_length=request.data_length,
      control_length=request.control_length,
      optimizer=request.optimizer,
      window_size=request.window_size,
      hidden_size=request.hidden_size,
      learning_rate=request.learning_rate,
      learning_rate_decrease_speed=request.learning_rate_decrease_speed,
      epochs=request.epochs,
      precision=request.precision
      )
   return response 

@router.post("/forecast")
def forecast_lstm_endpoint(request: LSTMForecastRequest):
   response = LSTMService.forecast_custom(
      csv_type=request.csv_type,
      column_name=request.column_name,
      data_length=request.data_length,
      control_length=request.control_length,
      optimizer=request.optimizer,
      window_size=request.window_size,
      hidden_size=request.hidden_size,
   )
   return response