# 🏗️ CareerShield: MLOps Architecture

This document visualizes the end-to-end lifecycle of the CareerShield layoff risk prediction platform, from code commit to real-time monitoring.

## System Workflow

```mermaid
graph TD
    %% Source Control
    A[GitHub Repository] -->|Push Code/Data| B(Jenkins CI/CD)

    %% Automation Layer
    subgraph CI_CD_Pipeline [Jenkins Automation]
        B --> C{Data Changed?}
        C -->|Yes| D[Model Retraining Stage]
        C -->|No| E[Skip Retraining]
        D --> F[Save New Artifacts]
        E --> G[Build Docker Images]
        F --> G
    end

    %% Artifact Storage
    G -->|Push Images| H[(Docker Hub)]

    %% Deployment Layer
    subgraph Kubernetes_Cluster [Minikube Cluster]
        H -->|Pull Images| I[K8s Deployments]
        I --> J[Frontend Pods x2]
        I --> K[Backend Pods x2]
        K -->|HPA| L{Scale Up/Down}
    end

    %% Monitoring Layer
    subgraph Monitoring_Stack [ELK Stack]
        K -->|Structured JSON| M[Logstash]
        M --> N[(Elasticsearch)]
        N --> O[Kibana Dashboard]
    end

    %% User Interaction
    P[End User] -->|HTTPS| J
    J -->|REST API| K
    O -->|Visualize Metrics| Q[Admin/Data Scientist]
```

## Key Technical Features

| Feature | Implementation |
|---------|----------------|
| **CI/CD** | Jenkins with conditional retraining logic |
| **Orchestration** | Kubernetes (Minikube) with Rolling Updates |
| **Scalability** | Horizontal Pod Autoscaler (HPA) based on CPU |
| **Observability** | ELK Stack with structured JSON logging & `/metrics` endpoint |
| **Model Serving** | FastAPI with TensorFlow backend |
| **Deployment** | Zero-downtime rolling updates |
