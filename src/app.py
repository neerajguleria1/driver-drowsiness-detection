from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import time
from contextlib import asynccontextmanager

from src.system_pipeline import DriverSafetySystem


# ------------------------
# LOGGING CONFIG
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------
# LIFESPAN (LOAD MODEL ONCE)
# ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Loading Driver Safety System...")

        app.state.system = DriverSafetySystem(
    model_path="models/final_driver_drowsiness_pipeline.pkl"
        )       

        app.state.system.health_check()

        logger.info("Driver Safety System loaded successfully ✅")

    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

    yield

    logger.info("Application shutting down...")


# ------------------------
# APP INIT
# ------------------------
app = FastAPI(
    title="Cognitive Driver Safety System",
    description="Production-grade decision intelligence API",
    version="1.0.0",
    lifespan=lifespan
)


# ------------------------
# SCHEMAS
# ------------------------
class DriverInput(BaseModel):
    Speed: float
    Alertness: float
    Seatbelt: int
    Heart_rate: float
    Fatigue: int
    speed_change: float
    prev_alertness: float


class Decision(BaseModel):
    action: str
    severity: str
    message: str


class DriverAnalysisResponse(BaseModel):
    ml_prediction: str
    ml_confidence: float
    risk_score: int
    risk_state: str
    decision: Decision
    explanations: List[str]


# ------------------------
# ROUTES
# ------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Driver Safety System is live 🚀"
    }


@app.post("/v1/analyze", response_model=DriverAnalysisResponse)
def analyze_driver(request: Request, input_data: DriverInput):

    system = request.app.state.system
    start_time = time.time()

    try:
        logger.info(
            f"Request received | speed={input_data.Speed} "
            f"fatigue={input_data.Fatigue}"
        )

        result = system.analyze(input_data.dict())

        latency = time.time() - start_time
        logger.info(f"Inference latency: {latency:.3f}s")

        return result

    # Client mistakes → 400
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Server failures → 500
    except Exception:
        logger.exception("Inference failed")

        raise HTTPException(
            status_code=500,
            detail="Internal inference error"
        )
