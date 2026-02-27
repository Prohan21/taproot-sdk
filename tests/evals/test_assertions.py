"""Tests for eval assertions."""

import pytest

from taproot_sdk.evals.assertions import assert_eval
from taproot_sdk.evals.exceptions import EvalAssertionError
from taproot_sdk.evals.models import AggregateScore, EvalResult


def _make_result(**overrides):
    defaults = {
        "run_id": "run-1",
        "status": "completed",
        "total_items": 10,
        "completed_items": 10,
        "failed_items": 2,
        "aggregate_scores": {
            "exact_match": AggregateScore(
                mean=0.8, min=0.0, max=1.0, std_dev=0.4, passed=8, failed=2,
            ),
        },
        "started_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:00:30Z",
    }
    defaults.update(overrides)
    return EvalResult(**defaults)


class TestAssertEval:
    def test_passing_result(self):
        result = _make_result()
        assert_eval(result, min_pass_rate=80)

    def test_failed_status_raises(self):
        result = _make_result(status="failed", error_message="boom")
        with pytest.raises(EvalAssertionError, match="did not complete"):
            assert_eval(result)

    def test_pass_rate_below_minimum(self):
        result = _make_result(failed_items=5)  # 50% pass rate
        with pytest.raises(EvalAssertionError, match="below minimum"):
            assert_eval(result, min_pass_rate=80)

    def test_min_score_passing(self):
        result = _make_result()
        assert_eval(result, min_score={"exact_match": 0.7})

    def test_min_score_failing(self):
        result = _make_result()
        with pytest.raises(EvalAssertionError, match="below minimum"):
            assert_eval(result, min_score={"exact_match": 0.9})

    def test_missing_metric_raises(self):
        result = _make_result()
        with pytest.raises(EvalAssertionError, match="not found"):
            assert_eval(result, min_score={"nonexistent": 0.5})

    def test_max_duration_passing(self):
        result = _make_result()
        assert_eval(result, max_duration_ms=60000)

    def test_max_duration_failing(self):
        result = _make_result()
        with pytest.raises(EvalAssertionError, match="exceeds maximum"):
            assert_eval(result, max_duration_ms=10000)  # 30s run > 10s max

    def test_all_assertions_combined(self):
        result = _make_result()
        assert_eval(
            result,
            min_pass_rate=70,
            min_score={"exact_match": 0.7},
            max_duration_ms=60000,
        )

    def test_error_has_result_attached(self):
        result = _make_result(status="failed")
        with pytest.raises(EvalAssertionError) as exc_info:
            assert_eval(result)
        assert exc_info.value.result is result
