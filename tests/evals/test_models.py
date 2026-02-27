"""Tests for eval models."""

from taproot_sdk.evals.models import AggregateScore, EvalResult, RunHandle


class TestRunHandle:
    def test_frozen(self):
        handle = RunHandle(run_id="abc", status="pending", message="queued")
        assert handle.run_id == "abc"
        assert handle.status == "pending"

    def test_immutable(self):
        handle = RunHandle(run_id="abc", status="pending", message="queued")
        try:
            handle.run_id = "xyz"  # type: ignore
            assert False, "Should raise"
        except AttributeError:
            pass


class TestEvalResult:
    def _make_result(self, **overrides):
        defaults = {
            "run_id": "run-1",
            "status": "completed",
            "total_items": 10,
            "completed_items": 10,
            "failed_items": 2,
        }
        defaults.update(overrides)
        return EvalResult(**defaults)

    def test_pass_rate(self):
        result = self._make_result(completed_items=10, failed_items=2)
        assert result.pass_rate == 80.0

    def test_pass_rate_zero_items(self):
        result = self._make_result(completed_items=0, failed_items=0)
        assert result.pass_rate == 0.0

    def test_duration_ms(self):
        result = self._make_result(
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:10Z",
        )
        assert result.duration_ms == 10000.0

    def test_duration_none_when_missing(self):
        result = self._make_result()
        assert result.duration_ms is None

    def test_from_api_response(self):
        data = {
            "id": "run-uuid",
            "status": "completed",
            "total_items": 5,
            "completed_items": 5,
            "failed_items": 1,
            "aggregate_scores": {
                "exact_match": {
                    "mean": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "std_dev": 0.4,
                    "passed": 4,
                    "failed": 1,
                }
            },
            "started_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T00:01:00Z",
            "error_message": None,
            "tags": ["ci"],
        }
        result = EvalResult.from_api_response(data)
        assert result.run_id == "run-uuid"
        assert result.status == "completed"
        assert "exact_match" in result.aggregate_scores
        assert result.aggregate_scores["exact_match"].mean == 0.8
        assert result.tags == ["ci"]

    def test_from_api_response_no_scores(self):
        data = {
            "id": "run-uuid",
            "status": "failed",
            "total_items": 5,
            "completed_items": 0,
            "failed_items": 0,
            "aggregate_scores": None,
        }
        result = EvalResult.from_api_response(data)
        assert result.aggregate_scores == {}
