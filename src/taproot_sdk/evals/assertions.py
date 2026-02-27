"""Eval assertions for CI/CD pipelines.

Usage in pytest:
    result = await client.wait_for_eval(run_id)
    assert_eval(result, min_pass_rate=80, min_score={"answer_relevancy": 0.7})
"""

from __future__ import annotations

from taproot_sdk.evals.exceptions import EvalAssertionError
from taproot_sdk.evals.models import EvalResult


def assert_eval(
    result: EvalResult,
    *,
    min_pass_rate: float | None = None,
    min_score: dict[str, float] | None = None,
    max_duration_ms: float | None = None,
) -> None:
    """Assert that an eval result meets the given criteria.

    Args:
        result: The completed eval result to check.
        min_pass_rate: Minimum pass rate percentage (0-100).
        min_score: Minimum mean score per metric name (0-1).
        max_duration_ms: Maximum run duration in milliseconds.

    Raises:
        EvalAssertionError: If any assertion fails.
    """
    if result.status != "completed":
        raise EvalAssertionError(
            f"Eval run did not complete successfully. Status: {result.status}, "
            f"Error: {result.error_message}",
            result,
        )

    if min_pass_rate is not None and result.pass_rate < min_pass_rate:
        raise EvalAssertionError(
            f"Pass rate {result.pass_rate:.1f}% is below minimum {min_pass_rate}%",
            result,
        )

    if min_score:
        for metric_name, threshold in min_score.items():
            score = result.aggregate_scores.get(metric_name)
            if score is None:
                raise EvalAssertionError(
                    f"Metric '{metric_name}' not found in results. "
                    f"Available: {list(result.aggregate_scores.keys())}",
                    result,
                )
            if score.mean < threshold:
                raise EvalAssertionError(
                    f"Metric '{metric_name}' mean score {score.mean:.3f} "
                    f"is below minimum {threshold}",
                    result,
                )

    if max_duration_ms is not None:
        duration = result.duration_ms
        if duration is not None and duration > max_duration_ms:
            raise EvalAssertionError(
                f"Run duration {duration:.0f}ms exceeds maximum {max_duration_ms}ms",
                result,
            )
