"""
Layoff Risk Prediction API
--------------------------
FastAPI inference endpoint for the MLOps Layoff Risk platform.

Endpoints:
  POST /predict          — single prediction
  POST /predict/batch    — batch predictions (up to 100 rows)
  GET  /health           — liveness probe
  GET  /model/info       — model metadata & schema
  GET  /industries       — list valid industries
  GET  /departments      — list valid departments

All requests are logged to stdout in ELK-compatible JSON (structured logging)
so Logstash can pick them up and feed the Kibana dashboard.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ─────────────────────────────────────────────
# Structured JSON logger (ELK-compatible)
# ─────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(payload)


def _build_logger(name: str) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


log = _build_logger("layoff_risk_api")


# ─────────────────────────────────────────────
# Load model & schema at startup
# ─────────────────────────────────────────────

MODELS_DIR  = os.getenv("MODELS_DIR", "./models")
MODEL_PATH  = os.path.join(MODELS_DIR, "layoff_risk_model.pkl")
SCHEMA_PATH = os.path.join(MODELS_DIR, "model_schema.json")

_pipeline: Any = None
_schema:   Dict[str, Any] = {}


def _load_artifacts() -> None:
    global _pipeline, _schema

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run the training notebook first to generate model artifacts."
        )

    _pipeline = joblib.load(MODEL_PATH)
    log.info("Model loaded", extra={"extra": {"model_path": MODEL_PATH}})

    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH) as fh:
            _schema = json.load(fh)
        log.info("Schema loaded", extra={"extra": {"schema_path": SCHEMA_PATH}})
    else:
        log.warning("Schema file not found — some metadata endpoints will be empty.")


# ─────────────────────────────────────────────
# Feature engineering helpers (mirrors notebook)
# ─────────────────────────────────────────────

_AI_MAP = {"No": 0, "Partial": 1, "Yes": 2}

_REASON_KEYWORDS: Dict[str, List[str]] = {
    "ai_automation":    ["ai", "automat"],
    "financial":        ["profit", "cost"],
    "restructuring":    ["restructur", "reorg"],
    "market_conditions":["market", "downturn"],
}


def _categorize_reason(reason: str) -> str:
    r = reason.lower()
    for category, keywords in _REASON_KEYWORDS.items():
        if any(kw in r for kw in keywords):
            return category
    return "other"


def _derive_workforce_band(total_employees: int) -> str:
    if total_employees < 1000:
        return "small"
    if total_employees < 5000:
        return "mid"
    if total_employees < 20000:
        return "large"
    return "enterprise"


def _build_feature_row(req: "PredictRequest") -> pd.DataFrame:
    """Convert a PredictRequest into a single-row DataFrame matching training features."""
    ai_num = _AI_MAP.get(req.ai_exposure, 0)
    industry_avg = _schema.get("industry_avg_pct_map", {}).get(req.industry, 10.0)
    median_open  = _schema.get("median_open_positions", 50.0)
    qmap         = _schema.get("quarter_risk_map", {})
    q_score      = float(qmap.get(str(req.quarter), 0.5))

    row = {
        # Numeric
        "Employees_Laid_Off":         req.employees_laid_off,
        "Severance_Weeks":            req.severance_weeks,
        "Total_Employees":            req.total_employees,
        "workforce_log":              float(np.log1p(req.total_employees)),
        "ai_exposure_num":            ai_num,
        "industry_avg_workforce_pct": industry_avg,
        "avg_open_positions":         median_open,
        "quarter_risk_score":         q_score,
        "Month":                      req.month,
        "Quarter":                    req.quarter,
        # Categorical
        "Industry":                   req.industry,
        "reason_category":            _categorize_reason(req.layoff_reason),
        "workforce_band":             _derive_workforce_band(req.total_employees),
        "primary_dept":               req.primary_department,
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input schema for a single layoff risk prediction."""

    industry: str = Field(
        ...,
        description="Company industry (e.g. 'Software', 'Fintech', 'EdTech')",
        examples=["Software"]
    )
    primary_department: str = Field(
        ...,
        description="Primary department affected (e.g. 'Engineering', 'Sales')",
        examples=["Engineering"]
    )
    ai_exposure: str = Field(
        ...,
        description="Company's AI adoption level: 'No', 'Partial', or 'Yes'",
        examples=["Partial"]
    )
    total_employees: int = Field(
        ..., gt=0,
        description="Total headcount at the company",
        examples=[5000]
    )
    employees_laid_off: int = Field(
        ..., ge=0,
        description="Number of employees being laid off",
        examples=[500]
    )
    severance_weeks: int = Field(
        default=8, ge=0, le=52,
        description="Severance package duration in weeks",
        examples=[8]
    )
    layoff_reason: str = Field(
        default="restructuring",
        description="Reason for the layoff event",
        examples=["AI replacing content creators"]
    )
    month: int = Field(
        default=1, ge=1, le=12,
        description="Month of the layoff (1–12)",
        examples=[1]
    )
    quarter: int = Field(
        default=1, ge=1, le=4,
        description="Quarter of the year (1–4)",
        examples=[1]
    )

    @field_validator("ai_exposure")
    @classmethod
    def validate_ai_exposure(cls, v: str) -> str:
        valid = {"No", "Partial", "Yes"}
        if v not in valid:
            raise ValueError(f"ai_exposure must be one of {valid}")
        return v

    @model_validator(mode="after")
    def employees_cannot_exceed_total(self) -> "PredictRequest":
        if self.employees_laid_off > self.total_employees:
            raise ValueError("employees_laid_off cannot exceed total_employees")
        return self


class RiskLevel(str):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class PredictResponse(BaseModel):
    request_id:        str
    timestamp:         str
    risk_label:        str        # HIGH / MEDIUM / LOW
    risk_probability:  float      # 0.0 – 1.0
    risk_score:        int        # 0 – 100
    percentage_workforce_affected: float
    top_risk_factors:  List[str]
    model_version:     str
    latency_ms:        float


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


# ─────────────────────────────────────────────
# Risk interpretation helpers
# ─────────────────────────────────────────────

def _risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    if prob >= 0.35:
        return "MEDIUM"
    return "LOW"


def _top_risk_factors(req: PredictRequest) -> List[str]:
    """Heuristic explanations surfaced alongside the model score."""
    factors: List[str] = []
    pct = (req.employees_laid_off / req.total_employees) * 100
    if pct > 15:
        factors.append(f"Large workforce reduction ({pct:.1f}% of headcount)")
    if req.ai_exposure == "Yes":
        factors.append("High AI automation exposure in the company")
    if req.quarter == 1:
        factors.append("Q1 — historically the highest layoff quarter")
    if req.severance_weeks <= 6:
        factors.append("Below-average severance indicates urgent cost cutting")
    if any(kw in req.layoff_reason.lower() for kw in ["ai", "automat"]):
        factors.append("Layoff reason linked to AI-driven role elimination")
    if not factors:
        factors.append("No dominant single risk factor — composite model signal")
    return factors[:4]


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Layoff Risk Prediction API",
    description=(
        "MLOps platform that predicts company layoff risk from industry, "
        "department, and AI exposure signals. Trained on live LinkedIn + news data."
    ),
    version="1.0.0",
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
    log.info("API startup complete")


# ─── Middleware: request tracing + ELK log ───

@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = str(uuid.uuid4())
    start      = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - start) * 1000
    log.info(
        "request",
        extra={
            "extra": {
                "request_id": request_id,
                "method":     request.method,
                "path":       request.url.path,
                "status":     response.status_code,
                "latency_ms": round(latency, 2),
            }
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness probe — used by Kubernetes / load balancers."""
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        model_version=_schema.get("best_model", "unknown"),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/model/info", tags=["Meta"])
def model_info() -> Dict[str, Any]:
    """Return model metadata, feature schema, and performance metrics."""
    if not _schema:
        raise HTTPException(status_code=503, detail="Schema not loaded.")
    return {
        "model":             _schema.get("best_model"),
        "test_auc":          _schema.get("test_auc"),
        "numeric_features":  _schema.get("numeric_features", []),
        "categorical_features": _schema.get("categorical_features", []),
        "risk_threshold_pct": _schema.get("risk_threshold"),
    }


@app.get("/industries", tags=["Meta"])
def list_industries() -> Dict[str, List[str]]:
    """Valid industry values accepted by the /predict endpoint."""
    return {"industries": _schema.get("industries", [])}


@app.get("/departments", tags=["Meta"])
def list_departments() -> Dict[str, List[str]]:
    """Valid department values accepted by the /predict endpoint."""
    return {"departments": _schema.get("departments", [])}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict layoff risk for a single company / department.

    Returns a risk label (LOW / MEDIUM / HIGH), probability score (0–1),
    and the top contributing risk factors.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    t0 = time.perf_counter()

    try:
        X = _build_feature_row(req)
        prob = float(_pipeline.predict_proba(X)[0][1])
    except Exception as exc:
        log.error("Prediction error", extra={"extra": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    latency = (time.perf_counter() - t0) * 1000
    pct_affected = round((req.employees_laid_off / req.total_employees) * 100, 2)

    response = PredictResponse(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat() + "Z",
        risk_label=_risk_label(prob),
        risk_probability=round(prob, 4),
        risk_score=int(prob * 100),
        percentage_workforce_affected=pct_affected,
        top_risk_factors=_top_risk_factors(req),
        model_version=_schema.get("best_model", "unknown"),
        latency_ms=round(latency, 2),
    )

    # ELK-structured prediction log (feeds Kibana dashboard)
    log.info(
        "prediction",
        extra={
            "extra": {
                "request_id":      response.request_id,
                "industry":        req.industry,
                "department":      req.primary_department,
                "ai_exposure":     req.ai_exposure,
                "risk_label":      response.risk_label,
                "risk_probability":response.risk_probability,
                "risk_score":      response.risk_score,
                "pct_affected":    pct_affected,
                "latency_ms":      response.latency_ms,
            }
        },
    )

    return response


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictRequest) -> BatchPredictResponse:
    """
    Predict layoff risk for up to 100 companies in a single request.

    Processes each row independently and returns results in the same order
    as the input list.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    results: List[PredictResponse] = []

    for req in batch.requests:
        try:
            X    = _build_feature_row(req)
            prob = float(_pipeline.predict_proba(X)[0][1])
        except Exception as exc:
            log.error("Batch item error", extra={"extra": {"error": str(exc)}})
            prob = 0.0  # degrade gracefully; surface error in risk_factors

        pct_affected = round((req.employees_laid_off / req.total_employees) * 100, 2)
        results.append(
            PredictResponse(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat() + "Z",
                risk_label=_risk_label(prob),
                risk_probability=round(prob, 4),
                risk_score=int(prob * 100),
                percentage_workforce_affected=pct_affected,
                top_risk_factors=_top_risk_factors(req),
                model_version=_schema.get("best_model", "unknown"),
                latency_ms=0.0,  # set at batch level
            )
        )

    total_latency = (time.perf_counter() - t0) * 1000
    log.info(
        "batch_prediction",
        extra={"extra": {"batch_size": len(results), "latency_ms": round(total_latency, 2)}},
    )

    return BatchPredictResponse(
        total=len(results),
        results=results,
        latency_ms=round(total_latency, 2),
    )


# ─────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
