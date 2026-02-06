from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import time
from contextlib import asynccontextmanager

from google.colab import files

#uploaded=files.upload()

from system_pipeline import DriverSafetySystem


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Global system reference
system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global system

    try:
        logger.info("Loading Driver Safety System...")
        system = DriverSafetySystem(
            model_path="final_driver_drowsiness_pipeline.pkl"
        )
        system.health_check()
        logger.info("System loaded successfully ✅")

    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

    yield

    logger.info("Application shutting down...")

app = FastAPI(
    title="Cognitive Driver Safety System",
    description="Production-grade decision intelligence API",
    version="1.0.0",
    lifespan=lifespan
)


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
    return {"status": "ok", "message": "Driver Safety System is live 🚀"}


@app.post("/v1/analyze", response_model=DriverAnalysisResponse)
def analyze_driver(input_data: DriverInput):

    if system is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    start_time = time.time()

    try:
        logger.info("Received analysis request")

        result = system.analyze(input_data.dict())

        latency = time.time() - start_time
        logger.info(f"Inference latency: {latency:.3f}s")

        return result

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail="Internal inference error"
        )
