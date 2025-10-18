from pydantic import BaseModel

class Metrics(BaseModel):
   MAE: float
   RMSE: float
   MAPE: float
