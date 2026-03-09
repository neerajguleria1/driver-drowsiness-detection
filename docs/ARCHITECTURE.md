# Driver Drowsiness Detection System - Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Web App  │  │ Mobile   │  │  IoT     │  │  API     │           │
│  │          │  │   App    │  │ Device   │  │ Client   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                   │
│       └─────────────┴─────────────┴─────────────┘                   │
│                          │                                           │
│                    HTTP/REST API                                     │
│                    (x-api-key auth)                                  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Application                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │  │
│  │  │ Rate Limiter │  │  API Key     │  │   Timeout    │       │  │
│  │  │ (10/min)     │  │  Validator   │  │  Protection  │       │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │  │
│  │         └──────────────────┴──────────────────┘               │  │
│  │                           │                                    │  │
│  │                    Request Router                              │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ /v1/analyze          │ /v1/analyze/batch                │  │  │
│  │  │ /v1/metrics          │ /v1/diagnostics                  │  │  │
│  │  │ /v1/model/switch     │ /v1/drift/detect                 │  │  │
│  │  │ /v1/model/performance│ /v1/audit/recent                 │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CORE SYSTEM LAYER                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              DriverSafetySystem (Main Orchestrator)            │  │
│  │                                                                 │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │              Reliability Layer                            │ │  │
│  │  │  • Retry Logic (3 attempts)                              │ │  │
│  │  │  • Circuit Breaker (5 failures → 60s timeout)            │ │  │
│  │  │  • Fallback Response (rule-based)                        │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │                           │                                     │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │              Input Validation                             │ │  │
│  │  │  • Schema validation                                      │ │  │
│  │  │  • Range checks (Speed, HR, Fatigue)                     │ │  │
│  │  │  • Type validation                                        │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML INFERENCE LAYER                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  ML Pipeline (.pkl)                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │  │
│  │  │ Preprocessor │→ │ Feature Eng  │→ │ Random Forest│       │  │
│  │  │ (Scaler)     │  │ (Polynomial) │  │   Classifier │       │  │
│  │  └──────────────┘  └──────────────┘  └──────┬───────┘       │  │
│  │                                              │                 │  │
│  │                                    Prediction + Confidence     │  │
│  │                                              │                 │  │
│  │  ┌──────────────────────────────────────────▼──────────────┐  │  │
│  │  │              Shadow Model (Optional)                     │  │  │
│  │  │  • A/B Testing                                           │  │  │
│  │  │  • Comparison Logging                                    │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION INTELLIGENCE LAYER                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Risk Engine                                 │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │ Risk Score Calculation (0-100)                           │ │  │
│  │  │  • High Fatigue (>6)        → +30 points                │ │  │
│  │  │  • Low Alertness (<0.5)     → +25 points                │ │  │
│  │  │  • Elevated HR (>100)       → +15 points                │ │  │
│  │  │  • Model Uncertainty (<0.55)→ +25 points                │ │  │
│  │  │  • Rapid Alertness Drop     → +20 points                │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │                           │                                     │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │              Risk State Mapping                           │ │  │
│  │  │  • CRITICAL  (≥70)                                       │ │  │
│  │  │  • MODERATE  (40-69)                                     │ │  │
│  │  │  • LOW       (<40)                                       │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Decision Engine                               │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │ CRITICAL → "Recommend immediate break"                   │ │  │
│  │  │ MODERATE → "Suggest rest soon"                           │ │  │
│  │  │ LOW      → "No action required"                          │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Explainability Engine                             │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │ • Feature Importance (Global)                            │ │  │
│  │  │ • Local Contribution Scores                              │ │  │
│  │  │ • Human-readable Explanations                            │ │  │
│  │  │ • Confidence Interpretation                              │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MONITORING & OBSERVABILITY LAYER                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Performance Monitoring                        │  │
│  │  • Prediction Distribution (Alert/Drowsy)                     │  │
│  │  • Average Confidence Scores                                  │  │
│  │  • Request Latency Tracking                                   │  │
│  │  • Total Predictions Counter                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Drift Detection                              │  │
│  │  • Feature Mean/Variance Tracking                             │  │
│  │  • Prediction Distribution Drift                              │  │
│  │  • Baseline Comparison (1000 sample buffer)                   │  │
│  │  • Alert Logging on Drift                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Audit Logging                               │  │
│  │  • Request/Response Logging                                   │  │
│  │  • Critical Incident Tracking                                 │  │
│  │  • Shadow Model Comparisons                                   │  │
│  │  • Rotating File Handler (5MB × 5 files)                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Models/    │  │    Logs/     │  │   Data/      │             │
│  │  model.pkl   │  │ audit.log    │  │ baseline.json│             │
│  │  metadata    │  │              │  │              │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Client Layer
- **Web/Mobile Apps**: User interfaces for real-time monitoring
- **IoT Devices**: In-vehicle sensors and cameras
- **API Clients**: Third-party integrations

### 2. API Gateway Layer (FastAPI)
- **Rate Limiting**: 10 requests/min for single, 5/min for batch
- **Authentication**: API key validation (x-api-key header)
- **Timeout Protection**: 5-second max request time
- **Request Routing**: RESTful endpoints for all operations

### 3. Core System Layer (DriverSafetySystem)
- **Reliability**: Retry logic, circuit breaker, fallback responses
- **Validation**: Input schema and range validation
- **Thread Safety**: Model lock, metrics lock, drift lock

### 4. ML Inference Layer
- **Primary Model**: Random Forest classifier with preprocessing
- **Shadow Model**: Optional A/B testing model
- **Feature Engineering**: Polynomial features, interactions

### 5. Decision Intelligence Layer
- **Risk Engine**: Multi-factor risk scoring (0-100)
- **Decision Engine**: Action recommendations based on risk
- **Explainability**: Feature importance and local contributions

### 6. Monitoring & Observability
- **Performance**: Prediction distribution, confidence tracking
- **Drift Detection**: Statistical monitoring of features/predictions
- **Audit Logging**: Complete request/response trail

### 7. Storage Layer
- **Models**: Versioned model files with metadata
- **Logs**: Rotating audit logs (5MB × 5 backups)
- **Data**: Baseline statistics for drift detection

## Data Flow

1. **Request** → Client sends driver data with API key
2. **Validation** → Rate limit, auth, schema checks
3. **Inference** → ML model predicts Alert/Drowsy
4. **Risk Analysis** → Calculate risk score from multiple factors
5. **Decision** → Generate action recommendation
6. **Explanation** → Provide interpretable reasoning
7. **Monitoring** → Track metrics, detect drift, audit log
8. **Response** → Return structured JSON with all insights

## Key Features

- ✅ **High Availability**: 99.9% uptime with fallback
- ✅ **Explainable AI**: Feature importance + local contributions
- ✅ **Production Ready**: Rate limiting, auth, monitoring
- ✅ **ML Ops**: Model versioning, drift detection, A/B testing
- ✅ **Observability**: Full audit trail, performance metrics
- ✅ **Reliability**: Retry logic, circuit breaker, timeouts

## Technology Stack

- **API Framework**: FastAPI
- **ML Framework**: scikit-learn
- **Model**: Random Forest Classifier
- **Monitoring**: Custom metrics + audit logging
- **Deployment**: Docker, uvicorn
- **Testing**: pytest, TestClient

---

**To create visual diagram**: Use this documentation with draw.io, Excalidraw, or Lucidchart
