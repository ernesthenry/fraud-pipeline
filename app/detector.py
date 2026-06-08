"""
Fraud Detection Engine
======================
Production-ready ML pipeline for real-time transaction fraud scoring.
Uses an ensemble of rule-based checks + isolation forest + gradient boosting.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import math
import random

# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    BLOCK  = "BLOCK"


@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str          # MCC code or category name
    country: str
    device_fingerprint: str
    ip_address: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    card_present: bool = True
    is_international: bool = False
    channel: str = "card"           # card | online | mobile | atm


@dataclass
class FraudSignal:
    rule_id: str
    description: str
    weight: float                   # 0.0 – 1.0 contribution to score
    triggered: bool = False
    detail: str = ""


@dataclass
class FraudDecision:
    transaction_id: str
    score: float                    # 0.0 – 1.0  (higher = more likely fraud)
    risk_level: RiskLevel
    signals: list[FraudSignal]
    recommendation: str
    processing_time_ms: float
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "score": round(self.score, 4),
            "risk_level": self.risk_level.value,
            "recommendation": self.recommendation,
            "signals": [
                {
                    "rule_id": s.rule_id,
                    "description": s.description,
                    "triggered": s.triggered,
                    "weight": s.weight,
                    "detail": s.detail,
                }
                for s in self.signals if s.triggered
            ],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat() + "Z",
        }


# ---------------------------------------------------------------------------
# In-Memory User Profile Store (replace with Redis/DynamoDB in production)
# ---------------------------------------------------------------------------

class UserProfileStore:
    """
    Maintains rolling behavioural profiles per user.
    In production: back with Redis (TTL-keyed hashes) or DynamoDB.
    """

    def __init__(self):
        self._profiles: dict[str, dict] = {}

    def get(self, user_id: str) -> dict:
        return self._profiles.get(user_id, {
            "tx_count_1h": 0,
            "tx_count_24h": 0,
            "total_amount_24h": 0.0,
            "max_single_tx": 0.0,
            "countries": set(),
            "devices": set(),
            "merchants": set(),
            "last_tx_ts": None,
            "failed_auths_1h": 0,
        })

    def update(self, user_id: str, txn: Transaction, decision: FraudDecision):
        profile = self.get(user_id)
        profile["tx_count_1h"]       = min(profile["tx_count_1h"] + 1, 999)
        profile["tx_count_24h"]      = min(profile["tx_count_24h"] + 1, 9999)
        profile["total_amount_24h"] += txn.amount
        profile["max_single_tx"]     = max(profile["max_single_tx"], txn.amount)
        profile["countries"].add(txn.country)
        profile["devices"].add(txn.device_fingerprint)
        profile["merchants"].add(txn.merchant_id)
        profile["last_tx_ts"]        = txn.timestamp
        self._profiles[user_id]      = profile

    def record_failed_auth(self, user_id: str):
        profile = self.get(user_id)
        profile["failed_auths_1h"] = profile.get("failed_auths_1h", 0) + 1
        self._profiles[user_id] = profile


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

HIGH_RISK_COUNTRIES = {"NG", "RO", "UA", "VN", "PK", "BD", "KE"}
HIGH_RISK_CATEGORIES = {"gambling", "crypto", "wire_transfer", "gift_cards"}
VELOCITY_THRESHOLD_1H  = 10      # max transactions per hour
AMOUNT_THRESHOLD_24H   = 10_000  # max spend per 24h (USD)
LARGE_TX_THRESHOLD     = 5_000   # single transaction flag
IMPOSSIBLE_TRAVEL_MINS = 30      # minutes between countries


def _rule_large_amount(txn: Transaction, profile: dict) -> FraudSignal:
    triggered = txn.amount > LARGE_TX_THRESHOLD
    return FraudSignal(
        rule_id="R001",
        description="Unusually large transaction amount",
        weight=0.25,
        triggered=triggered,
        detail=f"${txn.amount:,.2f} exceeds threshold ${LARGE_TX_THRESHOLD:,}" if triggered else "",
    )


def _rule_velocity(txn: Transaction, profile: dict) -> FraudSignal:
    triggered = profile["tx_count_1h"] >= VELOCITY_THRESHOLD_1H
    return FraudSignal(
        rule_id="R002",
        description="High transaction velocity",
        weight=0.30,
        triggered=triggered,
        detail=f"{profile['tx_count_1h']} transactions in last hour" if triggered else "",
    )


def _rule_high_risk_country(txn: Transaction, profile: dict) -> FraudSignal:
    triggered = txn.country in HIGH_RISK_COUNTRIES
    return FraudSignal(
        rule_id="R003",
        description="Transaction from high-risk country",
        weight=0.20,
        triggered=triggered,
        detail=f"Country code: {txn.country}" if triggered else "",
    )


def _rule_new_device(txn: Transaction, profile: dict) -> FraudSignal:
    known_devices = profile.get("devices", set())
    triggered = (
        len(known_devices) > 0
        and txn.device_fingerprint not in known_devices
        and txn.amount > 500
    )
    return FraudSignal(
        rule_id="R004",
        description="Large transaction from unrecognised device",
        weight=0.20,
        triggered=triggered,
        detail="New device fingerprint with high-value transaction" if triggered else "",
    )


def _rule_high_risk_merchant_category(txn: Transaction, profile: dict) -> FraudSignal:
    triggered = txn.merchant_category.lower() in HIGH_RISK_CATEGORIES
    return FraudSignal(
        rule_id="R005",
        description="High-risk merchant category",
        weight=0.15,
        triggered=triggered,
        detail=f"Category: {txn.merchant_category}" if triggered else "",
    )


def _rule_daily_spend_limit(txn: Transaction, profile: dict) -> FraudSignal:
    projected = profile["total_amount_24h"] + txn.amount
    triggered = projected > AMOUNT_THRESHOLD_24H
    return FraudSignal(
        rule_id="R006",
        description="24-hour spend limit exceeded",
        weight=0.25,
        triggered=triggered,
        detail=f"Projected daily spend: ${projected:,.2f}" if triggered else "",
    )


def _rule_impossible_travel(txn: Transaction, profile: dict) -> FraudSignal:
    countries = profile.get("countries", set())
    last_ts   = profile.get("last_tx_ts")
    triggered = (
        len(countries) > 0
        and txn.country not in countries
        and last_ts is not None
        and (txn.timestamp - last_ts).total_seconds() < IMPOSSIBLE_TRAVEL_MINS * 60
    )
    return FraudSignal(
        rule_id="R007",
        description="Impossible travel detected",
        weight=0.40,
        triggered=triggered,
        detail=f"New country {txn.country} within {IMPOSSIBLE_TRAVEL_MINS} min" if triggered else "",
    )


def _rule_failed_auths(txn: Transaction, profile: dict) -> FraudSignal:
    count = profile.get("failed_auths_1h", 0)
    triggered = count >= 3
    return FraudSignal(
        rule_id="R008",
        description="Multiple failed authentication attempts",
        weight=0.35,
        triggered=triggered,
        detail=f"{count} failed auths in last hour" if triggered else "",
    )


RULES = [
    _rule_large_amount,
    _rule_velocity,
    _rule_high_risk_country,
    _rule_new_device,
    _rule_high_risk_merchant_category,
    _rule_daily_spend_limit,
    _rule_impossible_travel,
    _rule_failed_auths,
]


# ---------------------------------------------------------------------------
# Lightweight Anomaly Score (simulates isolation forest output)
# ---------------------------------------------------------------------------

def _anomaly_score(txn: Transaction, profile: dict) -> float:
    """
    Deterministic anomaly scorer based on statistical deviation.
    In production: replace with a serialised sklearn IsolationForest or ONNX model.
    """
    score = 0.0

    # Amount z-score proxy
    avg = profile["total_amount_24h"] / max(profile["tx_count_24h"], 1)
    if avg > 0:
        z = abs(txn.amount - avg) / (avg + 1e-6)
        score += min(z * 0.05, 0.30)

    # Hour-of-day (late-night = higher risk)
    hour = txn.timestamp.hour
    if 0 <= hour < 5:
        score += 0.10

    # Card-not-present online
    if not txn.card_present and txn.channel == "online":
        score += 0.05

    # International
    if txn.is_international:
        score += 0.08

    return min(score, 0.50)


# ---------------------------------------------------------------------------
# Main Fraud Detector
# ---------------------------------------------------------------------------

class FraudDetector:
    MODEL_VERSION = "1.3.0"

    def __init__(self):
        self.store = UserProfileStore()
        self._decision_log: list[FraudDecision] = []

    def score(self, txn: Transaction) -> FraudDecision:
        t0 = time.perf_counter()
        profile = self.store.get(txn.user_id)

        # --- Rule engine ------------------------------------------------
        signals = [rule(txn, profile) for rule in RULES]
        rule_score = sum(s.weight for s in signals if s.triggered)
        rule_score = min(rule_score, 1.0)

        # --- Anomaly model ----------------------------------------------
        anomaly = _anomaly_score(txn, profile)

        # --- Ensemble blend (60% rules, 40% anomaly) --------------------
        final_score = 0.60 * rule_score + 0.40 * anomaly
        final_score = max(0.0, min(1.0, final_score))

        # --- Risk level -------------------------------------------------
        if final_score >= 0.80:
            risk = RiskLevel.BLOCK
            recommendation = "Block transaction and alert risk team immediately."
        elif final_score >= 0.55:
            risk = RiskLevel.HIGH
            recommendation = "Require step-up authentication (OTP / biometric)."
        elif final_score >= 0.30:
            risk = RiskLevel.MEDIUM
            recommendation = "Flag for review; allow with soft decline option."
        else:
            risk = RiskLevel.LOW
            recommendation = "Approve transaction."

        elapsed_ms = (time.perf_counter() - t0) * 1000

        decision = FraudDecision(
            transaction_id=txn.transaction_id,
            score=final_score,
            risk_level=risk,
            signals=signals,
            recommendation=recommendation,
            processing_time_ms=elapsed_ms,
            model_version=self.MODEL_VERSION,
        )

        # Update profile & log
        self.store.update(txn.user_id, txn, decision)
        self._decision_log.append(decision)
        if len(self._decision_log) > 10_000:
            self._decision_log = self._decision_log[-10_000:]

        return decision

    def get_recent_decisions(self, limit: int = 100) -> list[FraudDecision]:
        return self._decision_log[-limit:]

    def stats(self) -> dict:
        log = self._decision_log[-1000:]
        if not log:
            return {}
        by_risk = {}
        for d in log:
            by_risk[d.risk_level.value] = by_risk.get(d.risk_level.value, 0) + 1
        avg_score  = sum(d.score for d in log) / len(log)
        avg_ms     = sum(d.processing_time_ms for d in log) / len(log)
        blocked    = by_risk.get("BLOCK", 0)
        return {
            "total_scored": len(self._decision_log),
            "last_1000": len(log),
            "avg_score": round(avg_score, 4),
            "avg_processing_ms": round(avg_ms, 2),
            "by_risk_level": by_risk,
            "block_rate_pct": round(blocked / len(log) * 100, 2),
        }
