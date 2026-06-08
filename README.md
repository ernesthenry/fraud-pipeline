# Fraud Detection Pipeline

Production-ready real-time transaction fraud scoring service.

## Architecture

```
Client App
    │
    ▼
POST /v1/score          ← REST (single, <5ms)
POST /v1/score/batch    ← REST (up to 500 txns)
WS   /ws/stream         ← Real-time push to dashboards
    │
    ▼
Rule Engine (8 rules, weighted)
    +
Anomaly Scorer (statistical model)
    │
    ▼ Ensemble blend (60/40)
FraudDecision
    │
    ├── UserProfileStore (update rolling behavioural profile)
    └── WebSocket broadcast → dashboards
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (dev)
uvicorn app.api:app --reload --port 8000

# Run (prod, Docker)
docker-compose up --build

# API docs
open http://localhost:8000/docs
```

## Score a Transaction

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "amount": 4999.99,
    "currency": "USD",
    "merchant_id": "merch-xyz",
    "merchant_category": "online_retail",
    "country": "US",
    "device_fingerprint": "fp-abc",
    "ip_address": "1.2.3.4"
  }'
```

Response:
```json
{
  "transaction_id": "uuid",
  "score": 0.2100,
  "risk_level": "LOW",
  "recommendation": "Approve transaction.",
  "signals": [],
  "processing_time_ms": 0.18,
  "model_version": "1.3.0",
  "timestamp": "2026-01-01T00:00:00Z"
}
```

## Risk Levels

| Score      | Level  | Action                              |
|-----------|--------|-------------------------------------|
| 0.00–0.29 | LOW    | Approve                             |
| 0.30–0.54 | MEDIUM | Flag for review                     |
| 0.55–0.79 | HIGH   | Step-up authentication (OTP)        |
| 0.80–1.00 | BLOCK  | Block + alert risk team             |

## Rules

| ID   | Description                  | Weight |
|------|------------------------------|--------|
| R001 | Large transaction amount     | 0.25   |
| R002 | High velocity (≥10/hr)       | 0.30   |
| R003 | High-risk country            | 0.20   |
| R004 | New device + high amount     | 0.20   |
| R005 | High-risk merchant category  | 0.15   |
| R006 | 24-hour spend limit exceeded | 0.25   |
| R007 | Impossible travel            | 0.40   |
| R008 | Failed authentication burst  | 0.35   |

## Production Checklist

- [ ] Replace in-memory `UserProfileStore` with Redis (TTL-keyed hashes)
- [ ] Swap anomaly scorer for serialised sklearn `IsolationForest` / ONNX model
- [ ] Add JWT/API-key auth middleware
- [ ] Add structured logging (structlog / OpenTelemetry)
- [ ] Configure rate limiting (nginx / AWS WAF)
- [ ] Set up Prometheus metrics endpoint
- [ ] Enable TLS termination at load balancer
- [ ] Run `pytest tests/ -v` in CI before every deploy

## Tests

```bash
pytest tests/ -v
# 14 passed in < 1s
```
