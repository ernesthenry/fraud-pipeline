"""
Fraud Detection Pipeline – Test Suite
======================================
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.detector import (
    FraudDetector, Transaction, RiskLevel,
    LARGE_TX_THRESHOLD, VELOCITY_THRESHOLD_1H, HIGH_RISK_COUNTRIES
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tx(**overrides) -> Transaction:
    defaults = dict(
        transaction_id="tx-test-001",
        user_id="user-abc",
        amount=50.00,
        currency="USD",
        merchant_id="merch-001",
        merchant_category="grocery",
        country="US",
        device_fingerprint="fp-abc123",
        ip_address="192.168.1.1",
        card_present=True,
        is_international=False,
        channel="card",
    )
    defaults.update(overrides)
    return Transaction(**defaults)


# ---------------------------------------------------------------------------
# Tests: Score range
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_score_between_0_and_1(self):
        d = FraudDetector()
        decision = d.score(make_tx())
        assert 0.0 <= decision.score <= 1.0

    def test_low_risk_normal_transaction(self):
        d = FraudDetector()
        decision = d.score(make_tx(amount=25.00))
        assert decision.risk_level == RiskLevel.LOW, f"Expected LOW, got {decision.risk_level}"
        assert decision.score < 0.30

    def test_high_risk_large_amount(self):
        d = FraudDetector()
        decision = d.score(make_tx(amount=LARGE_TX_THRESHOLD + 1000))
        assert decision.score > 0.10  # at least rule R001 triggered

    def test_block_multiple_signals(self):
        d = FraudDetector()
        # Trigger velocity by scoring many transactions
        for i in range(VELOCITY_THRESHOLD_1H + 2):
            d.score(make_tx(transaction_id=f"tx-{i}", user_id="user-vel"))

        # Now score a suspicious one
        decision = d.score(make_tx(
            transaction_id="tx-final",
            user_id="user-vel",
            amount=LARGE_TX_THRESHOLD + 500,
            country=list(HIGH_RISK_COUNTRIES)[0],
        ))
        assert decision.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCK)


# ---------------------------------------------------------------------------
# Tests: Individual rules
# ---------------------------------------------------------------------------

class TestRules:
    def test_rule_r001_large_amount(self):
        d = FraudDetector()
        decision = d.score(make_tx(amount=LARGE_TX_THRESHOLD + 1))
        rule_ids = [s.rule_id for s in decision.signals if s.triggered]
        assert "R001" in rule_ids

    def test_rule_r003_high_risk_country(self):
        d = FraudDetector()
        decision = d.score(make_tx(country="NG"))
        rule_ids = [s.rule_id for s in decision.signals if s.triggered]
        assert "R003" in rule_ids

    def test_rule_r005_high_risk_merchant_category(self):
        d = FraudDetector()
        decision = d.score(make_tx(merchant_category="gambling"))
        rule_ids = [s.rule_id for s in decision.signals if s.triggered]
        assert "R005" in rule_ids

    def test_rule_r007_impossible_travel(self):
        d = FraudDetector()
        now = datetime.utcnow()
        # First tx from US
        d.score(make_tx(country="US", timestamp=now - timedelta(minutes=5)))
        # Second tx from NG 5 minutes later
        tx2 = make_tx(
            transaction_id="tx-travel",
            country="NG",
            timestamp=now,
        )
        decision = d.score(tx2)
        rule_ids = [s.rule_id for s in decision.signals if s.triggered]
        assert "R007" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Profile accumulation
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_profile_updated_after_score(self):
        d = FraudDetector()
        d.score(make_tx(user_id="user-profile", amount=200))
        profile = d.store.get("user-profile")
        assert profile["tx_count_1h"] == 1
        assert profile["total_amount_24h"] == 200.0

    def test_daily_limit_rule_r006(self):
        d = FraudDetector()
        uid = "user-daily"
        for i in range(5):
            d.score(make_tx(
                transaction_id=f"tx-d{i}",
                user_id=uid,
                amount=2_500,
            ))
        decision = d.score(make_tx(
            transaction_id="tx-d-over",
            user_id=uid,
            amount=500,
        ))
        rule_ids = [s.rule_id for s in decision.signals if s.triggered]
        assert "R006" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_scoring_under_50ms(self):
        import time
        d = FraudDetector()
        tx = make_tx()
        start = time.perf_counter()
        for _ in range(100):
            d.score(tx)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        assert elapsed < 50, f"Mean scoring time {elapsed:.1f} ms exceeds 50 ms"

    def test_batch_100_transactions(self):
        import time
        d = FraudDetector()
        txns = [make_tx(transaction_id=f"tx-batch-{i}", amount=float(i+1)) for i in range(100)]
        start = time.perf_counter()
        for t in txns:
            d.score(t)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 500, f"Batch of 100 took {elapsed:.1f} ms"


# ---------------------------------------------------------------------------
# Tests: Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_decision_dict_has_required_keys(self):
        d = FraudDetector()
        result = d.score(make_tx()).to_dict()
        for key in ["transaction_id", "score", "risk_level", "recommendation",
                    "signals", "processing_time_ms", "model_version", "timestamp"]:
            assert key in result, f"Missing key: {key}"

    def test_stats_structure(self):
        d = FraudDetector()
        d.score(make_tx())
        s = d.stats()
        assert "total_scored" in s
        assert "avg_score" in s
        assert "by_risk_level" in s
