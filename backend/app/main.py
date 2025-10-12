from fastapi import FastAPI
from app.api.v1 import lstm

app = FastAPI(title="Custom LSTM API")

app.include_router(lstm.router, prefix="/api/v1/lstm", tags=["LSTM"])
