from fastapi import APIRouter

from backend.app.services.portfolio_service import PortfolioService

router = APIRouter()

@router.post("/optimize")
def markowitz_optimization_endpoint(request: OptimizationRequest):
   response = PortfolioService.optimize(

   )

   return response