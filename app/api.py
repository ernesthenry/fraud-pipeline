"""
Fraud Detection API
===================
FastAPI server exposing:
  POST /v1/score            – score a single transaction (< 20 ms p99)
  POST /v1/score/batch      – score up to 500 transactions
  GET  /v1/health           – liveness + model version
  GET  /v1/stats            – live detection statistics
  GET  /v1/decisions        – recent decision log
  WS   /ws/stream           – real-time scored-transaction stream

Usage:
  pip install fastapi uvicorn pydantic
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from app.detector import FraudDetector, Transaction, RiskLevel

# ---------------------------------------------------------------------------
# App Bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time ML-powered transaction fraud scoring pipeline",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FraudDetector()

# WebSocket connection pool
_ws_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str                  = Field(..., min_length=1, max_length=128)
    amount: float                 = Field(..., gt=0, le=1_000_000)
    currency: str                 = Field("USD", min_length=3, max_length=3)
    merchant_id: str              = Field(..., min_length=1)
    merchant_category: str        = Field(..., min_length=1)
    country: str                  = Field(..., min_length=2, max_length=2)
    device_fingerprint: str       = Field(..., min_length=1)
    ip_address: str               = Field(..., min_length=7)
    card_present: bool            = True
    is_international: bool        = False
    channel: str                  = Field("card", pattern="^(card|online|mobile|atm)$")

    @validator("amount")
    def amount_precision(cls, v):
        return round(v, 2)


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_items=1, max_items=500)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_start_time = datetime.utcnow()


def _to_domain(req: TransactionRequest) -> Transaction:
    return Transaction(
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        amount=req.amount,
        currency=req.currency,
        merchant_id=req.merchant_id,
        merchant_category=req.merchant_category,
        country=req.country,
        device_fingerprint=req.device_fingerprint,
        ip_address=req.ip_address,
        card_present=req.card_present,
        is_international=req.is_international,
        channel=req.channel,
    )


async def _broadcast(payload: dict):
    """Push a scored decision to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    return HealthResponse(
        status="ok",
        model_version=FraudDetector.MODEL_VERSION,
        uptime_seconds=round(uptime, 1),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/v1/stats", tags=["Analytics"])
async def stats():
    return JSONResponse(content=detector.stats())


@app.get("/v1/decisions", tags=["Analytics"])
async def recent_decisions(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be 1–500")
    decisions = detector.get_recent_decisions(limit)
    return JSONResponse(content={
        "count": len(decisions),
        "decisions": [d.to_dict() for d in reversed(decisions)],
    })


@app.post("/v1/score", tags=["Scoring"])
async def score_transaction(req: TransactionRequest, background_tasks: BackgroundTasks):
    """
    Score a single transaction.
    Returns fraud score, risk level, triggered signals, and recommendation.
    """
    txn      = _to_domain(req)
    decision = detector.score(txn)
    payload  = decision.to_dict()

    # Non-blocking broadcast to WebSocket listeners
    background_tasks.add_task(_broadcast, payload)

    return JSONResponse(
        content=payload,
        status_code=200,
    )


@app.post("/v1/score/batch", tags=["Scoring"])
async def score_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    """
    Score up to 500 transactions in a single request.
    Responses preserve input order.
    """
    results = []
    for item in req.transactions:
        txn      = _to_domain(item)
        decision = detector.score(txn)
        results.append(decision.to_dict())

    background_tasks.add_task(
        _broadcast, {"batch": True, "count": len(results), "decisions": results}
    )

    return JSONResponse(content={
        "count": len(results),
        "results": results,
    })


# ---------------------------------------------------------------------------
# WebSocket – Real-time stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Connect to receive every scored transaction as it happens.
    Clients receive JSON payloads matching the /v1/score response schema.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep alive — ignore incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )
