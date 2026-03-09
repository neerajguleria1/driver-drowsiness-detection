from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import logging
import time
import uuid
import os
import asyncio
import pandas as pd
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.system_pipeline import DriverSafetySystem
from src.cv_api import router as cv_router


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

API_KEY = os.getenv("API_KEY", "dev_secure_key_123")


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
# APP INIT (CRITICAL)
# ------------------------
app = FastAPI(
    title="Cognitive Driver Safety System",
    description="Production-grade explainable decision intelligence API",
    version="2.1.0",
    lifespan=lifespan
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Include Computer Vision Router
app.include_router(cv_router)


# ------------------------
# GLOBAL ERROR HANDLER
# ------------------------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

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
# API KEY VERIFICATION
# ------------------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


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


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    global_importance: float
    local_contribution_score: float


class DriverAnalysisResponse(BaseModel):
    ml_prediction: str
    ml_confidence: float
    confidence_level: str
    risk_score: int
    risk_state: str
    risk_factors: List[str]
    decision: Decision
    top_contributing_features: List[FeatureContribution]
    explanations: List[str]
    model_version: str
    model_type: str
    inference_latency_ms: float
    trace_id: str
    fallback_mode: bool = False
    error: str = None


# ------------------------
# HEALTH ENDPOINT
# ------------------------
@app.get("/health")
def deep_health(request: Request):

    try:
        request.app.state.system.health_check()
        return {"status": "healthy"}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="System unhealthy"
        )


# ------------------------
# MAIN INFERENCE ROUTE
# ------------------------
@app.post("/v1/analyze", response_model=DriverAnalysisResponse)
@limiter.limit("10/minute")
async def analyze_driver(
    request: Request,
    input_data: DriverInput,
    api_key: str = Depends(verify_api_key)
):

    system = request.app.state.system

    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Timeout protection: 5 seconds max
        result = await asyncio.wait_for(
            asyncio.to_thread(
                system.analyze,
                input_data.model_dump(),
                trace_id
            ),
            timeout=5.0
        )

        total_latency = time.time() - start_time
        logger.info(f"[{trace_id}] total_request_latency: {total_latency:.3f}s")

        result["trace_id"] = trace_id
        return result

    except asyncio.TimeoutError:
        logger.error(f"[{trace_id}] Request timeout")
        raise HTTPException(status_code=504, detail="Request timeout")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"[{trace_id}] Inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error")        
# ------------------------
# BATCH ENDPOINT
# ------------------------
@app.post("/v1/analyze/batch")
@limiter.limit("5/minute")
async def analyze_batch(
    request: Request,
    inputs: List[DriverInput],
    api_key: str = Depends(verify_api_key)
):
    if len(inputs) > 200:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large"
        )

    system = request.app.state.system

    results = system.analyze_batch(
        [item.model_dump() for item in inputs]
    )

    return {
        "total_processed": len(results),
        "results": results
    }

@app.get("/v1/metrics")
def metrics(request: Request):
    return request.app.state.system.get_metrics()


# ------------------------
# DIAGNOSTICS ENDPOINT
# ------------------------
@app.get("/v1/diagnostics")
def diagnostics(request: Request):
    return request.app.state.system.diagnostics()

@app.get("/v1/audit/recent")
def get_recent_audit(limit: int = 10):

    import json

    try:
        with open("logs/driver_audit.log", "r") as f:
            lines = f.readlines()[-limit:]
            records = [json.loads(line) for line in lines]
    except FileNotFoundError:
        return {"message": "No audit logs yet"}

    return {
        "count": len(records),
        "records": records
    }


# ------------------------
# MODEL VERSIONING ENDPOINTS
# ------------------------
@app.post("/v1/model/switch/{version}")
def switch_model(version: str, request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request.app.state.system.load_model_version(version)
        return {"message": f"Switched to model {version}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/model/shadow/{version}")
def load_shadow(version: str, request: Request, api_key: str = Depends(verify_api_key)):
    try:
        request.app.state.system.load_shadow_model(version)
        return {"message": f"Shadow model {version} loaded for testing"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/model/shadow/disable")
def disable_shadow(request: Request, api_key: str = Depends(verify_api_key)):
    request.app.state.system.shadow_model = None
    return {"message": "Shadow model disabled"}


# ------------------------
# DRIFT DETECTION ENDPOINT
# ------------------------
@app.get("/v1/drift/detect")
def detect_drift(request: Request, api_key: str = Depends(verify_api_key)):
    return request.app.state.system.detect_drift()

@app.post("/v1/drift/baseline")
def set_baseline(request: Request, api_key: str = Depends(verify_api_key)):
    system = request.app.state.system
    if len(system.live_feature_buffer) < 100:
        raise HTTPException(status_code=400, detail="Insufficient data for baseline")
    
    baseline_df = pd.DataFrame(system.live_feature_buffer)[system.MODEL_FEATURES]
    system.set_baseline_stats(baseline_df, system.prediction_buffer)
    return {"message": "Baseline statistics set"}


# ------------------------
# PERFORMANCE MONITORING
# ------------------------
@app.get("/v1/model/performance")
def model_performance(request: Request, api_key: str = Depends(verify_api_key)):
    return request.app.state.system.performance_report()
