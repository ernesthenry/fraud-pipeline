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

# Import required libraries
import asyncio  # For asynchronous operations (handling multiple tasks concurrently)
import uuid  # For generating unique transaction IDs
from datetime import datetime  # For timestamps
from typing import Optional  # For optional type hints

# FastAPI imports - these are the core components for building the API
from fastapi import FastAPI  # The main application class
from fastapi import HTTPException  # For raising HTTP errors
from fastapi import WebSocket, WebSocketDisconnect  # For real-time connections
from fastapi import BackgroundTasks  # For running tasks in the background
from fastapi.middleware.cors import CORSMiddleware  # For handling cross-origin requests
from fastapi.responses import JSONResponse  # For sending JSON responses
from pydantic import BaseModel, Field, validator  # For data validation and schemas

# Import our custom fraud detection logic
from app.detector import FraudDetector, Transaction, RiskLevel

# ---------------------------------------------------------------------------
# App Bootstrap - Setting up the FastAPI application
# ---------------------------------------------------------------------------

# Create the FastAPI application instance
# This is the main object that handles all HTTP requests
app = FastAPI(
    title="Fraud Detection API",  # Name shown in API documentation
    description="Real-time ML-powered transaction fraud scoring pipeline",  # API description
    version="1.3.0",  # Current version of the API
    docs_url="/docs",  # URL for interactive API documentation (Swagger UI)
    redoc_url="/redoc",  # URL for alternative documentation (ReDoc)
)

# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows the API to be called from different domains/websites
# In production, you should restrict allow_origins to specific trusted domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (change this in production!)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Create an instance of our fraud detector
# This object contains all the logic for analyzing transactions
detector = FraudDetector()

# WebSocket connection pool
# This list keeps track of all currently connected WebSocket clients
# When a transaction is scored, we'll broadcast the result to all connected clients
_ws_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Pydantic Schemas - Data models for request/response validation
# ---------------------------------------------------------------------------

# TransactionRequest defines what data we expect when someone wants to score a transaction
# Pydantic automatically validates that incoming data matches this structure
class TransactionRequest(BaseModel):
    # Optional: if not provided, we'll generate a unique ID automatically
    transaction_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Required fields (marked with ... means mandatory)
    user_id: str = Field(..., min_length=1, max_length=128)  # ID of the user making the transaction
    amount: float = Field(..., gt=0, le=1_000_000)  # Transaction amount (must be positive, max 1M)
    currency: str = Field("USD", min_length=3, max_length=3)  # Currency code (default: USD)
    merchant_id: str = Field(..., min_length=1)  # ID of the merchant
    merchant_category: str = Field(..., min_length=1)  # Type of merchant (e.g., "online_retail")
    country: str = Field(..., min_length=2, max_length=2)  # 2-letter country code (e.g., "US")
    device_fingerprint: str = Field(..., min_length=1)  # Unique identifier for the device
    ip_address: str = Field(..., min_length=7)  # IP address of the request
    
    # Optional fields with default values
    card_present: bool = True  # Whether the physical card was present
    is_international: bool = False  # Whether this is an international transaction
    channel: str = Field("card", pattern="^(card|online|mobile|atm)$")  # Transaction channel

    # Custom validator: ensures amount is rounded to 2 decimal places (like currency)
    @validator("amount")
    def amount_precision(cls, v):
        return round(v, 2)


# BatchRequest for scoring multiple transactions at once
class BatchRequest(BaseModel):
    # List of transactions to score (1 to 500 at a time)
    transactions: list[TransactionRequest] = Field(..., min_items=1, max_items=500)


# HealthResponse defines the structure of the health check endpoint response
class HealthResponse(BaseModel):
    status: str  # "ok" if the service is running
    model_version: str  # Version of the fraud detection model
    uptime_seconds: float  # How long the service has been running
    timestamp: str  # Current time in ISO format


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

# Record when the application started (for uptime calculation)
_start_time = datetime.utcnow()


# Convert the API request model to the internal domain model
# This separates the external API interface from our internal business logic
def _to_domain(req: TransactionRequest) -> Transaction:
    """Convert TransactionRequest (API model) to Transaction (internal model)."""
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


# Broadcast a fraud decision to all connected WebSocket clients
# This allows real-time dashboards to receive updates as transactions are scored
async def _broadcast(payload: dict):
    """Push a scored decision to all connected WebSocket clients."""
    dead = []  # Track disconnected clients to remove them
    
    # Send the payload to each connected WebSocket client
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)  # Send JSON data to the client
        except Exception:
            # If sending fails, the client is likely disconnected
            dead.append(ws)
    
    # Remove disconnected clients from the pool
    for ws in dead:
        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# API Routes - Endpoints that clients can call
# ---------------------------------------------------------------------------

# Health check endpoint - used to verify the service is running
# Called by monitoring systems and load balancers
@app.get("/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the service is healthy and running."""
    # Calculate how long the service has been running
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return HealthResponse(
        status="ok",  # Service is healthy
        model_version=FraudDetector.MODEL_VERSION,  # Version of the fraud model
        uptime_seconds=round(uptime, 1),  # Uptime in seconds
        timestamp=datetime.utcnow().isoformat() + "Z",  # Current timestamp
    )


# Statistics endpoint - returns live detection statistics
# Shows how many transactions have been scored and their risk distribution
@app.get("/v1/stats", tags=["Analytics"])
async def stats():
    """Get live fraud detection statistics."""
    return JSONResponse(content=detector.stats())


# Recent decisions endpoint - returns a log of recent fraud decisions
# Useful for auditing and reviewing past transaction scores
@app.get("/v1/decisions", tags=["Analytics"])
async def recent_decisions(limit: int = 50):
    """Get recent fraud decisions (most recent first)."""
    # Validate the limit parameter
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be 1–500")
    
    # Get recent decisions from the detector
    decisions = detector.get_recent_decisions(limit)
    
    return JSONResponse(content={
        "count": len(decisions),  # Number of decisions returned
        "decisions": [d.to_dict() for d in reversed(decisions)],  # Convert to dict, newest first
    })


# Score a single transaction - the main fraud detection endpoint
# This is what client applications will call to check if a transaction is fraudulent
@app.post("/v1/score", tags=["Scoring"])
async def score_transaction(req: TransactionRequest, background_tasks: BackgroundTasks):
    """
    Score a single transaction for fraud risk.
    Returns fraud score, risk level, triggered signals, and recommendation.
    """
    # Convert the API request to our internal transaction model
    txn = _to_domain(req)
    
    # Run the fraud detection logic
    decision = detector.score(txn)
    
    # Convert the decision to a dictionary for JSON response
    payload = decision.to_dict()

    # Broadcast the result to WebSocket clients in the background
    # This doesn't block the response - it happens asynchronously
    background_tasks.add_task(_broadcast, payload)

    # Return the fraud decision to the caller
    return JSONResponse(
        content=payload,
        status_code=200,
    )


# Score multiple transactions at once - batch processing endpoint
# More efficient than calling /v1/score multiple times
@app.post("/v1/score/batch", tags=["Scoring"])
async def score_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    """
    Score up to 500 transactions in a single request.
    Responses preserve input order.
    """
    results = []
    
    # Process each transaction in the batch
    for item in req.transactions:
        txn = _to_domain(item)  # Convert to internal model
        decision = detector.score(txn)  # Score for fraud
        results.append(decision.to_dict())  # Add to results

    # Broadcast the batch results to WebSocket clients
    background_tasks.add_task(
        _broadcast, {"batch": True, "count": len(results), "decisions": results}
    )

    # Return all results
    return JSONResponse(content={
        "count": len(results),
        "results": results,
    })


# ---------------------------------------------------------------------------
# WebSocket Endpoint - Real-time streaming
# ---------------------------------------------------------------------------

# WebSocket endpoint for real-time fraud decision streaming
# Dashboard applications can connect here to receive live updates
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Connect to receive every scored transaction as it happens.
    Clients receive JSON payloads matching the /v1/score response schema.
    """
    # Accept the WebSocket connection
    await websocket.accept()
    
    # Add this client to our connection pool
    _ws_clients.append(websocket)
    
    try:
        # Keep the connection alive and listen for messages
        while True:
            # We don't expect clients to send meaningful data
            # This just keeps the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Client disconnected - remove them from the pool
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Error Handlers - Graceful error handling
# ---------------------------------------------------------------------------

# Global exception handler - catches any unhandled errors
# This prevents the API from crashing and returns a proper error response
@app.exception_handler(Exception)
async def generic_error(request, exc):
    """Handle any uncaught exceptions gracefully."""
    return JSONResponse(
        status_code=500,  # Internal Server Error
        content={"error": "internal_error", "detail": str(exc)},
    )
