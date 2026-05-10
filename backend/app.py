"""
Layoff Risk Prediction API  v3.0 (TensorFlow)
---------------------------------------------
FastAPI inference endpoint for the MLOps Layoff Risk platform — TensorFlow backend.

Request  POST /predict:
    { "industry", "department", "ai_exposure", "total_employees" }

Response:
    { request_id, timestamp, risk_probability, risk_score, risk_label,
      impact_level, top_risk_factors, career_advice, model_version, latency_ms }

Feature engineering is IDENTICAL to notebook Stage 5 — both load the same
model weights, preprocessor, and model_schema.json from the models/ directory.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# Global Observability Counters
# ─────────────────────────────────────────────────────────────────────────────
_TOTAL_PREDICTIONS = 0
_TOTAL_LATENCY_MS  = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Structured JSON logger  — ELK / Logstash compatible
# ─────────────────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level":     record.levelname,
            "message":   record.getMessage(),
            "logger":    record.name,
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 1. Console Handler (for kubectl logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_JsonFormatter())
    logger.addHandler(console_handler)
    
    # 2. Logstash UDP Handler (for ELK)
    logstash_host = os.getenv("LOGSTASH_HOST")
    if logstash_host:
        try:
            from logging.handlers import DatagramHandler
            # Logstash is listening on UDP 12201
            udp_handler = DatagramHandler(logstash_host, 12201)
            udp_handler.setFormatter(_JsonFormatter())
            logger.addHandler(udp_handler)
        except Exception as e:
            print(f"Failed to initialize Logstash handler: {e}")
            
    return logger


log = _build_logger("layoff_risk_api")


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loader  — Keras model + preprocessor + schema from disk
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR         = os.getenv("MODELS_DIR", "./models")
MODEL_PATH         = os.path.join(MODELS_DIR, "layoff_risk_model.keras")
PREPROCESSOR_PATH  = os.path.join(MODELS_DIR, "preprocessor.pkl")
SCHEMA_PATH        = os.path.join(MODELS_DIR, "model_schema.json")

_MODEL:        keras.Model        = None
_PREPROCESSOR: Any                = None
_schema:       Dict[str, Any]    = {}

_AI_MAP:       Dict[str, int]    = {}
_INDUSTRY_AVG: Dict[str, float]  = {}
_QUARTER_RISK: Dict[str, float]  = {}
_BAND_BREAKS:  List[int]         = []
_BAND_LABELS:  List[str]         = []
_MEDIAN_OPEN:  float             = 50.0


def _load_artifacts() -> None:
    global _MODEL, _PREPROCESSOR, _schema
    global _AI_MAP, _INDUSTRY_AVG, _QUARTER_RISK, _BAND_BREAKS, _BAND_LABELS, _MEDIAN_OPEN

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run training notebook first to generate artifacts."
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. "
            "Run training notebook first."
        )
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(
            f"Schema not found at {SCHEMA_PATH}. "
            "Run training notebook first."
        )

    _MODEL = keras.models.load_model(MODEL_PATH)
    _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    with open(SCHEMA_PATH) as fh:
        _schema = json.load(fh)

    _AI_MAP       = _schema["ai_exposure_map"]
    _INDUSTRY_AVG = _schema["industry_avg_pct_map"]
    _QUARTER_RISK = _schema["quarter_risk_map"]
    _BAND_BREAKS  = _schema["workforce_band_breaks"]
    _BAND_LABELS  = _schema["workforce_band_labels"]
    _MEDIAN_OPEN  = float(_schema["median_open_positions"])

    log.info("Artifacts loaded", extra={"extra": {
        "model":      _schema["model_version"],
        "test_auc":   _schema["test_auc"],
        "industries": len(_INDUSTRY_AVG),
    }})


# ─────────────────────────────────────────────────────────────────────────────
# Static lookup tables  — response enrichment, not model features
# ─────────────────────────────────────────────────────────────────────────────

_CAREER_MAP: Dict[tuple, tuple] = {
    ("Software",          "Engineering"):     ("AI Engineer",              5, "$160,000"),
    ("Software",          "Sales"):           ("Technical Sales Manager",  4, "$130,000"),
    ("Software",          "Marketing"):       ("Product Marketing Lead",   3, "$120,000"),
    ("Software",          "Operations"):      ("DevOps Engineer",          6, "$140,000"),
    ("Software",          "Content"):         ("AI Content Strategist",    2, "$115,000"),
    ("Software",          "Customer Support"):("AI Prompt Engineer",       3, "$105,000"),
    ("Fintech",           "Engineering"):     ("Blockchain Developer",     6, "$155,000"),
    ("Fintech",           "Sales"):           ("Fintech Solutions Lead",   4, "$125,000"),
    ("Fintech",           "Operations"):      ("Risk Automation Analyst",  4, "$130,000"),
    ("EdTech",            "Engineering"):     ("ML Engineer",              5, "$145,000"),
    ("EdTech",            "Marketing"):       ("AI Content Strategist",    2, "$110,000"),
    ("Semiconductors",    "Engineering"):     ("AI Hardware Engineer",     8, "$175,000"),
    ("Semiconductors/AI", "Engineering"):     ("AI Chip Architect",        9, "$185,000"),
    ("SaaS/CRM",          "Engineering"):     ("Platform Engineer",        4, "$150,000"),
    ("SaaS/CRM",          "Sales"):           ("Solutions Engineer",       3, "$140,000"),
    ("Gaming/Software",   "Engineering"):     ("Game AI Developer",        5, "$145,000"),
    ("Gaming/Software",   "Marketing"):       ("Growth Marketing Manager", 3, "$115,000"),
    ("Social Media",      "Engineering"):     ("ML Personalization Eng",   6, "$165,000"),
    ("Social Media",      "Content"):         ("AI Content Strategist",    2, "$110,000"),
    ("Consulting",        "Operations"):      ("Strategy & Ops Manager",   3, "$135,000"),
    ("Consulting",        "Consulting"):      ("AI Strategy Consultant",   4, "$145,000"),
    ("Cryptocurrency",    "Engineering"):     ("Web3 Engineer",            5, "$150,000"),
    ("Automotive/EV",     "Engineering"):     ("Autonomous Systems Eng",   7, "$170,000"),
    ("Automotive/EV",     "Manufacturing"):   ("Robotics Engineer",        5, "$155,000"),
    ("E-commerce",        "Operations"):      ("Supply Chain Analyst",     3, "$120,000"),
    ("E-commerce",        "Engineering"):     ("Recommender Systems Eng",  5, "$148,000"),
    ("E-commerce/Cloud",  "Engineering"):     ("Cloud Commerce Engineer",  4, "$150,000"),
    ("Streaming",         "Engineering"):     ("Video Infra Engineer",     4, "$148,000"),
    ("Search/AI",         "Engineering"):     ("LLM Engineer",             5, "$175,000"),
    ("Hardware",          "Engineering"):     ("Embedded AI Engineer",     6, "$160,000"),
    ("Networking",        "Engineering"):     ("Network Automation Eng",   5, "$145,000"),
    ("Cloud Storage",     "Engineering"):     ("Cloud Infra Engineer",     4, "$152,000"),
    ("Live Streaming",    "Engineering"):     ("Real-Time Systems Eng",    5, "$148,000"),
    ("Travel Tech",       "Engineering"):     ("Travel AI Analyst",        4, "$138,000"),
    ("Travel Tech",       "Operations"):      ("Revenue Management Analyst",3,"$125,000"),
    ("Communication",     "Engineering"):     ("Messaging Platform Eng",   4, "$145,000"),
    ("Music Streaming",   "Engineering"):     ("Audio ML Engineer",        5, "$148,000"),
    ("Video Conferencing","Engineering"):     ("WebRTC Engineer",          4, "$148,000"),
    ("Ridesharing",       "Engineering"):     ("Geo-ML Engineer",          5, "$152,000"),
    ("Fitness Tech",      "Engineering"):     ("Health AI Engineer",       5, "$142,000"),
}

_DEFAULT_DEPT_CAREERS: Dict[str, tuple] = {
    "Sales":            ("AI Sales Strategist", 3, "$135,000"),
    "Marketing":        ("Market AI Director", 3, "$125,000"),
    "Engineering":      ("AI Solutions Engineer", 5, "$150,000"),
    "Operations":       ("AI Operations Architecture", 4, "$135,000"),
    "Support":          ("Support Automation Lead", 3, "$110,000"),
    "Customer Support": ("Support Automation Lead", 3, "$110,000"),
    "Content":          ("AI Content Strategist", 2, "$105,000"),
    "Manufacturing":    ("Robotics/AI Integrator", 6, "$140,000"),
    "Consulting":       ("AI Transformation Lead", 4, "$150,000"),
    "IT Services":      ("AI Integration Specialist", 4, "$135,000"),
}
_GLOBAL_DEFAULT_CAREER = ("AI Integration Specialist", 4, "$130,000")

_INDUSTRY_MSG: Dict[str, str] = {
    "high":   "Industry has a high recent layoff rate",
    "medium": "Industry shows moderate layoff trends",
    "low":    "Industry has been relatively stable",
}
_DEPT_MSG: Dict[str, str] = {
    "Engineering":      "Engineering roles are relatively stable",
    "Sales":            "Sales headcount is often first to be cut in downturns",
    "Marketing":        "Marketing budgets are an early cost-reduction target",
    "Operations":       "Operations roles face automation-driven reduction risk",
    "Manufacturing":    "Manufacturing is highly exposed to automation",
    "Support":          "Support roles have high AI replacement risk",
    "Content":          "Content roles are increasingly automated by AI",
    "Customer Support": "Customer support is highly exposed to AI automation",
    "Consulting":       "Consulting headcount shrinks post-restructuring",
    "IT Services":      "IT services face both outsourcing and automation pressure",
    "AWS":              "Cloud infrastructure roles remain in high demand",
    "Azure":            "Cloud infrastructure roles remain in high demand",
    "Cloud":            "Cloud roles are relatively protected",
    "Reality Labs":     "XR/metaverse investments are being scaled back industry-wide",
}
_DEFAULT_DEPT_MSG = "Department shows average stability in current market"

_SIZE_MSG: Dict[str, str] = {
    "small":      "Small company size increases individual role risk",
    "mid":        "Mid-size company — moderate structural risk",
    "large":      "Large company size reduces but does not eliminate risk",
    "enterprise": "Enterprise scale reduces risk of total role elimination",
}
_AI_MSG: Dict[str, str] = {
    "Yes":     "High AI adoption signals role automation risk",
    "Partial": "Partial AI adoption may trigger selective automation cuts",
    "No":      "Low AI exposure limits automation-driven layoff risk",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — IDENTICAL to notebook Stage 5
# ─────────────────────────────────────────────────────────────────────────────

def _workforce_band(n: int) -> str:
    if n < _BAND_BREAKS[1]: return _BAND_LABELS[0]
    if n < _BAND_BREAKS[2]: return _BAND_LABELS[1]
    if n < _BAND_BREAKS[3]: return _BAND_LABELS[2]
    return _BAND_LABELS[3]


def _build_feature_row(industry: str, department: str,
                       ai_exposure: str, total_employees: int) -> pd.DataFrame:
    ai_num   = _AI_MAP.get(ai_exposure, 0)
    ind_avg  = _INDUSTRY_AVG.get(industry, 10.0)
    q_score  = float(_QUARTER_RISK.get("1", 0.5))
    est_laid = max(1, int(total_employees * ind_avg / 100))
    band     = _workforce_band(total_employees)

    return pd.DataFrame([{
        "Employees_Laid_Off":          est_laid,
        "Severance_Weeks":             8,
        "Total_Employees":             total_employees,
        "workforce_log":               math.log1p(total_employees),
        "ai_exposure_num":             ai_num,
        "industry_avg_workforce_pct":  ind_avg,
        "avg_open_positions":          _MEDIAN_OPEN,
        "quarter_risk_score":          q_score,
        "Month":                       1,
        "Quarter":                     1,
        "Industry":                    industry,
        "reason_category":             "restructuring",
        "workforce_band":              band,
        "primary_dept":                department,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Response enrichment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _risk_label(prob: float) -> str:
    if prob >= 0.65: return "HIGH"
    if prob >= 0.35: return "MEDIUM"
    return "LOW"


def _impact_level(prob: float) -> str:
    if prob >= 0.65: return "SEVERE"
    if prob >= 0.45: return "MODERATE"
    if prob >= 0.25: return "LOW"
    return "MINIMAL"


def _top_risk_factors(industry: str, department: str,
                      ai_exposure: str, total_employees: int) -> List[str]:
    ind_avg  = _INDUSTRY_AVG.get(industry, 10.0)
    ind_tier = "high" if ind_avg >= 15 else ("medium" if ind_avg >= 7 else "low")
    return [
        _INDUSTRY_MSG[ind_tier],
        _DEPT_MSG.get(department, _DEFAULT_DEPT_MSG),
        _SIZE_MSG[_workforce_band(total_employees)],
        _AI_MSG.get(ai_exposure, "AI exposure impact is unclear"),
    ]


def _career_advice(industry: str, department: str) -> Dict[str, Any]:
    # Search for an exact match ignoring case
    for (ik, dk), values in _CAREER_MAP.items():
        if ik.lower() == industry.lower() and dk.lower() == department.lower():
            return {"target_role": values[0], "time_months": values[1], "salary": values[2]}
            
    # Default to department-specific roles
    for dk, values in _DEFAULT_DEPT_CAREERS.items():
        if isinstance(values, tuple) and dk.lower() == department.lower():
            return {"target_role": values[0], "time_months": values[1], "salary": values[2]}
            
    # Absolute fallback
    role, months, salary = _GLOBAL_DEFAULT_CAREER
    return {"target_role": role, "time_months": months, "salary": salary}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    industry: str = Field(..., examples=["Software"])
    department: str = Field(..., examples=["Engineering"])
    ai_exposure: str = Field(..., examples=["Partial"])
    total_employees: int = Field(..., gt=0, examples=[5000])

    @field_validator("ai_exposure")
    @classmethod
    def validate_ai_exposure(cls, v: str) -> str:
        if v not in {"No", "Partial", "Yes"}:
            raise ValueError("ai_exposure must be 'No', 'Partial', or 'Yes'")
        return v


class CareerAdvice(BaseModel):
    target_role: str
    time_months: int
    salary:      str


class PredictResponse(BaseModel):
    request_id:       str
    timestamp:        str
    risk_probability: float
    risk_score:       int
    risk_label:       str
    impact_level:     str
    top_risk_factors: List[str]
    career_advice:    CareerAdvice
    model_version:    str
    latency_ms:       float


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest] = Field(..., max_length=100)


class BatchPredictResponse(BaseModel):
    total:      int
    results:    List[PredictResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    timestamp:     str


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function — TensorFlow backend
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(industry: str, department: str,
                  ai_exposure: str, total_employees: int) -> PredictResponse:
    """
    Pure inference — no FastAPI dependency.
    Uses Keras model + sklearn preprocessor.
    """
    t0    = time.perf_counter()
    X_row = _build_feature_row(industry, department, ai_exposure, total_employees)
    
    # Preprocess → dense numpy → predict
    X_processed = np.array(_PREPROCESSOR.transform(X_row))
    prob        = float(_MODEL.predict(X_processed, verbose=0).ravel()[0])
    
    ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        request_id=       str(uuid.uuid4()),
        timestamp=        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        risk_probability= round(prob, 4),
        risk_score=       int(prob * 100),
        risk_label=       _risk_label(prob),
        impact_level=     _impact_level(prob),
        top_risk_factors= _top_risk_factors(industry, department, ai_exposure, total_employees),
        career_advice=    CareerAdvice(**_career_advice(industry, department)),
        model_version=    _schema["model_version"],
        latency_ms=       ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Layoff Risk Prediction API",
    description=(
        "MLOps platform predicting company layoff risk from industry, "
        "department, and AI exposure. TensorFlow backend. "
        "All logs feed ELK → Kibana."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    _load_artifacts()
    log.info("API ready", extra={"extra": {"version": "3.0.0"}})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid   = str(uuid.uuid4())
    start = time.perf_counter()
    resp  = await call_next(request)
    ms    = round((time.perf_counter() - start) * 1000, 2)
    log.info("http", extra={"extra": {
        "event": "http_request",
        "request_id": rid, "method": request.method,
        "path": request.url.path, "status": resp.status_code, "latency_ms": ms,
    }})
    resp.headers["X-Request-ID"] = rid
    return resp


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness probe."""
    log.info("health_check", extra={"extra": {"event": "health_check"}})
    return HealthResponse(
        status="ok", model_loaded=_MODEL is not None,
        model_version=_schema.get("model_version", "unknown"),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@app.get("/metrics", tags=["Observability"])
def get_metrics() -> Dict[str, Any]:
    """Expose internal performance metrics."""
    avg_latency = 0.0
    if _TOTAL_PREDICTIONS > 0:
        avg_latency = round(_TOTAL_LATENCY_MS / _TOTAL_PREDICTIONS, 2)
    
    return {
        "event":              "metrics_export",
        "total_predictions":  _TOTAL_PREDICTIONS,
        "average_latency_ms": avg_latency,
        "model_version":      _schema.get("model_version", "unknown"),
        "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/model/info", tags=["Meta"])
def model_info() -> Dict[str, Any]:
    """Model metadata and test performance."""
    if not _schema:
        raise HTTPException(status_code=503, detail="Schema not loaded.")
    return {
        "model_version":        _schema.get("model_version"),
        "best_model":           _schema.get("best_model"),
        "test_auc":             _schema.get("test_auc"),
        "test_ap":              _schema.get("test_ap"),
        "risk_threshold_pct":   _schema.get("risk_threshold"),
        "score_threshold":      _schema.get("score_threshold"),
        "numeric_features":     _schema.get("numeric_features", []),
        "categorical_features": _schema.get("categorical_features", []),
    }


@app.get("/industries", tags=["Meta"])
def list_industries() -> Dict[str, List[str]]:
    return {"industries": sorted(_INDUSTRY_AVG.keys())}


@app.get("/departments", tags=["Meta"])
def list_departments() -> Dict[str, List[str]]:
    return {"departments": sorted(_DEPT_MSG.keys())}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict layoff risk for a single company.

    Input:
    ```json
    { "industry": "Software", "department": "Engineering",
      "ai_exposure": "Partial", "total_employees": 5000 }
    ```
    """
    if _MODEL is None or _PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        response = run_inference(
            req.industry, req.department, req.ai_exposure, req.total_employees
        )
    except Exception as exc:
        log.error("predict_error", extra={"extra": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    global _TOTAL_PREDICTIONS, _TOTAL_LATENCY_MS
    _TOTAL_PREDICTIONS += 1
    _TOTAL_LATENCY_MS  += response.latency_ms

    log.info("prediction", extra={"extra": {
        "event":            "prediction_event",
        "request_id":       response.request_id,
        "industry":         req.industry,
        "department":       req.department,
        "ai_exposure":      req.ai_exposure,
        "total_employees":  req.total_employees,
        "risk_label":       response.risk_label,
        "impact_level":     response.impact_level,
        "risk_probability": response.risk_probability,
        "risk_score":       response.risk_score,
        "latency_ms":       response.latency_ms,
    }})
    return response


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictRequest) -> BatchPredictResponse:
    """Predict layoff risk for up to 100 companies in one call."""
    if _MODEL is None or _PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0      = time.perf_counter()
    results: List[PredictResponse] = []

    for req in batch.requests:
        try:
            results.append(run_inference(
                req.industry, req.department, req.ai_exposure, req.total_employees
            ))
        except Exception as exc:
            log.error("batch_item_error", extra={"extra": {"error": str(exc)}})
            results.append(PredictResponse(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                risk_probability=0.0, risk_score=0, risk_label="LOW",
                impact_level="MINIMAL", top_risk_factors=[f"Error: {exc}"],
                career_advice=CareerAdvice(target_role="N/A", time_months=0, salary="N/A"),
                model_version=_schema.get("model_version", "unknown"), latency_ms=0.0,
            ))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("batch", extra={"extra": {"batch_size": len(results), "latency_ms": total_ms}})
    return BatchPredictResponse(total=len(results), results=results, latency_ms=total_ms)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)