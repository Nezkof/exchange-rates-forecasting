from fastapi import APIRouter

from app.schemas.portfolio import OptimizationRequest
from app.services.portfolio_service import PortfolioService

router = APIRouter()

@router.post("/optimize")
def markowitz_optimization_endpoint(request: OptimizationRequest):
   print(request)
   response = PortfolioService.optimize(
      csv_type = request.csv_type,
      tickers = request.tickers,
      data_length = request.data_length,
      control_length = request.control_length,
      optimizer = request.optimizer,
      window_size = request.window_size,
      hidden_size = request.hidden_size,
      samples_amount = request.samples_amount,
      risk_threshold = request.risk_threshold,
      capital = request.capital
   )

   return response