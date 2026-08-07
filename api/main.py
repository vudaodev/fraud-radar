from fastapi import FastAPI, Request
from common.paths import MODEL
from common.schemas import Transaction, ScoreResponse
from common.anomaly import FRAUD_THRESHOLD
from api.scoring import load_model, score
from contextlib import asynccontextmanager

# Define startup and shutdown logic with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(MODEL)
    yield

# Create an instance and assign it to 'app'
app = FastAPI(lifespan=lifespan)

@app.post("/score", response_model= ScoreResponse)
def post_score(transaction: Transaction, request: Request):
    s, f = score(request.app.state.model, transaction)
    return {"score": s, 
            "flagged": f}

@app.get("/health")
def check_health(request: Request):
    return {"model_loaded": hasattr(request.app.state, "model"),
            "threshold":FRAUD_THRESHOLD}

