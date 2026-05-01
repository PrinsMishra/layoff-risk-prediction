
# 🛡️ CareerShield: MLOps Layoff Risk Platform

**End-to-end MLOps platform** predicting company layoff risk from industry, department, AI exposure, and workforce size. Features a **React/Vite Frontend**, **FastAPI Backend**, **TensorFlow** models, and real-time telemetry with the **ELK Stack**, all containerized with **Docker Compose**.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org)
[![ELK](https://img.shields.io/badge/ELK_Stack-8.11-yellow.svg)](https://elastic.co)

---

## 📁 Project Structure

```
layoff-risk-prediction/
├── frontend/                      # React/Vite UI Application
│   ├── src/                       # React components & styles
│   ├── Dockerfile                 # Frontend multi-stage build (Nginx)
│   └── nginx.conf                 # Nginx configuration
├── elk/                           # ELK Telemetry Stack
│   └── logstash.conf              # Logstash UDP pipeline config
├── models/                        # Trained artifacts & charts
│   ├── layoff_risk_model.keras    # TensorFlow model
│   └── preprocessor.pkl           # Sklearn preprocessing pipeline
├── app.py                         # FastAPI inference server
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Backend Docker build
├── docker-compose.yml             # Full-stack orchestration
└── mlops-dataset-layoff-risk/     # Raw dataset
```

---

## 🚀 Quick Start

### Full Stack (Recommended)

The easiest way to run the entire platform (Frontend, Backend, and ELK Stack) is using Docker Compose:

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f
```

Once running, access the services at:
- 🌐 **Frontend UI**: [http://localhost:80](http://localhost:80)
- 🔌 **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📊 **Kibana Dashboards**: [http://localhost:5601](http://localhost:5601)

### Local Development (Backend only)

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost (TensorFlow equivalent) |
| **Test ROC-AUC** | 0.9234 |
| **Test AP** | 0.8912 |
| **Epochs** | Up to 100 (early stopping) |
| **CV Folds** | 5-fold stratified |

### Model Comparison
![Model Comparison](models/model_comparison.png)

### Learning Curves
![Learning Curves](models/learning_curves.png)

### Validation Evaluation
![Validation Curves](models/validation_curves.png)

### Feature Importance
![Feature Importance](models/feature_importance.png)

### Threshold Analysis
![Threshold Analysis](models/threshold_analysis.png)

### Test Set Evaluation
![Test Evaluation](models/test_evaluation.png)

### Probability Calibration
![Calibration](models/calibration.png)

### Sample Inference Results
![Inference Results](models/inference_results.png)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model metadata & performance |
| `GET` | `/industries` | List valid industries |
| `GET` | `/departments` | List valid departments |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch prediction (max 100) |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "industry": "Software",
    "department": "Engineering",
    "ai_exposure": "Partial",
    "total_employees": 5000
  }'
```

### Example Response

```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-04-25T09:36:00Z",
  "risk_probability": 0.4231,
  "risk_score": 42,
  "risk_label": "MEDIUM",
  "impact_level": "LOW",
  "top_risk_factors": [
    "Industry shows moderate layoff trends",
    "Engineering roles are relatively stable",
    "Large company size reduces but does not eliminate risk",
    "Partial AI adoption may trigger selective automation cuts"
  ],
  "career_advice": {
    "target_role": "AI Engineer",
    "time_months": 5,
    "salary": "$160,000"
  },
  "model_version": "XGBoost_tf_v1",
  "latency_ms": 12.34
}
```

---

## 🐳 Docker Architecture

The project uses `docker-compose` to orchestrate 5 connected services:

1. **frontend**: Serves the compiled React UI via Nginx on port `80`.
2. **backend**: FastAPI inference server on port `8000`, running the TensorFlow model. Sends logs to Logstash via the GELF driver.
3. **elasticsearch**: Stores and indexes inference logs and application telemetry.
4. **logstash**: Receives GELF logs from the backend via UDP `12201` and forwards them to Elasticsearch.
5. **kibana**: Visualizes logs and inference metrics on port `5601`.

### Managing Services

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs for a specific service
docker-compose logs -f frontend

# Rebuild a specific service (e.g. backend) after code changes
docker-compose up -d --build backend
```

---

## 📋 Requirements

- **Python** 3.12+
- **TensorFlow** 2.16+
- **FastAPI** 0.110+
- **Docker** 24.0+ (optional)

---

## 📝 License

MIT License — see repository for details.
```