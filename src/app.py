from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import logging
import time
import uuid
import os
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

from src.system_pipeline import DriverSafetySystem


# ------------------------
# LOGGING CONFIG
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------
# ENV CONFIG
# ------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/final_driver_drowsiness_pipeline.pkl"
)


# ------------------------
# LIFESPAN (LOAD MODEL ONCE)
# ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Loading Driver Safety System...")

        app.state.system = DriverSafetySystem(
            model_path=MODEL_PATH
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
# GLOBAL CRASH HANDLER
# ------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    trace_id = str(uuid.uuid4())

    logger.exception(f"[{trace_id}] Unhandled crash: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "trace_id": trace_id
        }
    )


# ------------------------
# SCHEMAS
# ------------------------
class DriverInput(BaseModel):
    Speed: float = Field(..., ge=0, le=200)
    Alertness: float = Field(..., ge=0.0, le=1.0)
    Seatbelt: int = Field(..., ge=0, le=1)
    HR: float = Field(..., ge=30, le=200)
    Fatigue: int = Field(..., ge=0, le=10)
    speed_change: float = Field(..., ge=0)
    prev_alertness: float = Field(..., ge=0.0, le=1.0)


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
    trace_id: str


# ------------------------
# HEALTH ROUTES
# ------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Driver Safety System is live 🚀"
    }


@app.get("/health")
def deep_health(request: Request):

    try:
        request.app.state.system.health_check()
        return {"status": "healthy"}

    except Exception as e:
        logger.critical(f"Health check failed: {e}")

        raise HTTPException(
            status_code=503,
            detail="System unhealthy"
        )


# ------------------------
# MAIN INFERENCE ROUTE
# ------------------------
@app.post("/v1/analyze", response_model=DriverAnalysisResponse)
def analyze_driver(request: Request, input_data: DriverInput):

    system = request.app.state.system
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        logger.info(
            f"[{trace_id}] Request received | "
            f"speed={input_data.Speed} fatigue={input_data.Fatigue}"
        )

        result = system.analyze(input_data.model_dump())

        latency = time.time() - start_time

        if latency > 1:
            logger.warning(f"[{trace_id}] HIGH latency: {latency:.3f}s")
        else:
            logger.info(f"[{trace_id}] latency: {latency:.3f}s")

        result["trace_id"] = trace_id
        return result

    except ValueError as e:
        logger.warning(f"[{trace_id}] Validation error: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception(f"[{trace_id}] Inference failed")

        raise HTTPException(
            status_code=500,
            detail="Internal inference error"
        )
