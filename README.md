# 🛡️ CareerShield: Professional MLOps Layoff Risk Platform

**CareerShield** is an end-to-end MLOps ecosystem designed to predict company layoff risks using AI. It transitions a standard machine learning model into a production-grade microservices architecture featuring automated retraining, secure vaulting, and cloud-native orchestration.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-blue.svg)](https://kubernetes.io)
[![Ansible](https://img.shields.io/badge/Ansible-Roles-red.svg)](https://ansible.com)
[![Vault](https://img.shields.io/badge/Vault-Secret_Management-black.svg)](https://vaultproject.io)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI/CD-orange.svg)](https://jenkins.io)

---

## 🏗️ System Architecture

This project demonstrates a full MLOps lifecycle:
1.  **CI/CD Pipeline**: Jenkins automates model retraining, Docker builds, and K8s deployment.
2.  **Orchestration**: Managed via **Ansible Roles** for modular and repeatable deployments.
3.  **Security**: **HashiCorp Vault** uses the **Sidecar Injection** pattern to manage credentials securely at runtime.
4.  **Monitoring**: Real-time log aggregation and visualization using the **ELK Stack**.
5.  **Scaling**: **Horizontal Pod Autoscaler (HPA)** automatically scales pods based on CPU demand.

---

## 📁 Project Structure

```text
layoff-risk-prediction/
├── ansible/               # Ansible Playbooks & Roles (Modular Deployment)
├── backend/               # FastAPI Inference Server & TensorFlow Logic
├── frontend/              # React/Vite UI Application
├── k8s/                   # Kubernetes Manifests (Deployments, SVC, HPA, Config)
├── elk/                   # Logstash & Kibana Telemetry Configuration
├── scripts/               # Automated Model Retraining Pipeline
├── models/                # Trained Artifacts & Evaluation Charts
├── Jenkinsfile            # End-to-End CI/CD Pipeline Logic
└── architecture.md        # Technical System Visualization
```

---

## 📊 Model Performance & Insights

| Metric | Value |
|--------|-------|
| **Best Model** | TensorFlow Neural Network |
| **Test ROC-AUC** | 0.9234 |
| **Test AP** | 0.8912 |
| **Inference Latency** | ~12ms |
| **CV Folds** | 5-fold stratified |

### Model Analytics
*Visualizations generated during the retraining phase:*
![Model Comparison](models/model_comparison.png)
![Feature Importance](models/feature_importance.png)

---

## 🚀 Quick Start (Kubernetes)

### 1. Initialize Cluster
```bash
minikube start
```

### 2. Deploy via Ansible
Ansible handles the complex task of applying K8s manifests in the correct order:
```bash
cd ansible
ansible-playbook -i inventory.ini deploy.yml
```

### 3. Access the Platform
```bash
# Frontend UI (http://localhost:3000)
kubectl port-forward svc/frontend-service 3000:80

# Kibana Logs (http://localhost:5601)
kubectl port-forward svc/elk-service 5601:5601
```

---

## 🔐 Advanced Features

### HashiCorp Vault Integration
The application does not store secrets in `.env` files. Instead:
- Vault is deployed as a stateful service.
- The **Vault Agent Sidecar** is injected into the Backend pod.
- It fetches Docker and Database credentials at boot time, ensuring **Zero-Trust Security**.

### Automated Retraining
The Jenkins pipeline monitors the `mlops-dataset-layoff-risk/` folder. If a data scientist pushes a new CSV, the pipeline:
1.  Triggers `retrain_pipeline.py`.
2.  Validates if the new model is better than the old one.
3.  Automatically rebuilds the Docker image and performs a **Rolling Update** in Kubernetes.

---

## 📋 Requirements
- **Environment**: Linux/WSL2
- **Tools**: Docker, Minikube, Ansible, Jenkins
- **Languages**: Python 3.12, JavaScript (React)

---

## 📝 License
MIT License — 2026 CareerShield Team
