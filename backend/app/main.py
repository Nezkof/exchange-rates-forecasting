from fastapi import FastAPI
from app.api.v1 import lstm
from app.api.v1 import portfolio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Custom LSTM API")

origins = [
   "http://localhost:3000",
]

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],  
   allow_headers=["*"], 
)

app.include_router(lstm.router, prefix="/api/v1/lstm", tags=["LSTM"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
