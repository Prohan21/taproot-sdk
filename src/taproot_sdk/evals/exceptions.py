"""Eval SDK exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taproot_sdk.evals.models import EvalResult


class EvalAssertionError(AssertionError):
    """Raised when an eval assertion fails.

    Designed for pytest integration — provides a clear failure message
    with the actual vs expected values.
    """

    def __init__(self, message: str, result: EvalResult) -> None:
        self.result = result
        super().__init__(message)
