# Deployment Guide

## Local Runtime

Use Docker Compose to run the API, MLflow, Prometheus, and Grafana together:

```bash
docker compose up --build
```

The default service endpoints are:

- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Production Readiness Notes

- Replace the local JSON model registry with a managed registry such as MLflow Model Registry, S3-backed artifact storage, or Azure ML.
- Move MLflow artifact storage from the local filesystem to cloud object storage.
- Run the API behind an ingress or API gateway with TLS termination.
- Externalize secrets through GitHub Actions secrets, Azure Key Vault, AWS Secrets Manager, or Vault.
- Use a managed scheduler or container platform for Airflow, Prefect workers, and inference services.

## Cloud Deployment Patterns

### Option 1: Kubernetes-Based Platform

- Run the inference API as a Deployment behind an Ingress.
- Run Prometheus and Grafana either in-cluster or as managed observability services.
- Use Argo CD, Flux, or GitHub Actions for deployment promotion.
- Host MLflow on a dedicated stateful workload with external object storage and relational metadata store.

### Option 2: Managed Container Platform

- Deploy the API to Azure Container Apps, AWS ECS Fargate, or Google Cloud Run.
- Offload monitoring to managed Prometheus and managed dashboards.
- Keep orchestration in managed Airflow or Prefect Cloud while inference remains separately deployable.

### Option 3: Hybrid Batch and Online Pattern

- Serve the champion model online through FastAPI.
- Run retraining and batch scoring on a scheduled compute platform.
- Persist batch outputs to an analytical table or object storage for downstream activation.

## Suggested Cloud Deployment Topology

1. Build and push the API image via GitHub Actions.
2. Deploy the inference service to Kubernetes, Azure Container Apps, ECS, or Cloud Run.
3. Persist monitoring data in managed Prometheus and managed Grafana.
4. Publish training artifacts to remote MLflow tracking and artifact stores.
5. Route scheduled retraining through Airflow or Prefect workers in the same environment.

## Rollback Strategy

The repository models rollback as a registry-level champion reassignment.

- Keep the current production model as the champion alias.
- Register newly trained candidates as challengers by default.
- Promote challengers only after offline validation and deployment checks.
- If production errors or KPI degradation occur, point the champion alias back to the previous known-good version.

This mirrors how real platforms reduce deployment risk: promotion changes traffic routing metadata rather than rebuilding models under pressure.