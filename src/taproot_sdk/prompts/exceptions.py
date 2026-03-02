"""Custom exceptions for the prompts module."""

from __future__ import annotations

from taproot_sdk.exceptions import TaprootError


class MissingVariableError(TaprootError):
    """Raised when a required template variable is not provided during rendering.

    Inherits from ``TaprootError`` so ``except TaprootError`` catches all
    SDK errors including missing variables.

    Attributes:
        variable_name: The name of the missing variable.
        required_variables: The full list of required variables for the prompt.
    """

    def __init__(
        self,
        variable_name: str,
        required_variables: tuple[str, ...],
    ) -> None:
        self.variable_name = variable_name
        self.required_variables = required_variables
        super().__init__(
            f"Missing required variable '{variable_name}'. "
            f"Required variables: {list(required_variables)}"
        )
