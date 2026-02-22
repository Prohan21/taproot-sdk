"""Data models for prompt responses.

PromptResponse is a frozen dataclass representing a prompt template
fetched from the Taproot serving layer. It supports client-side variable
rendering and content integrity verification.
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any

from taproot_sdk.prompts.exceptions import MissingVariableError

logger = logging.getLogger(__name__)

# Pattern for {{variable_name}} placeholders
_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class PromptResponse:
    """Immutable representation of a prompt fetched from the serving layer.

    Attributes:
        schema_version: API schema version (currently 1).
        name: The prompt name identifier.
        version: The resolved version number.
        content: The raw template content with {{variable}} placeholders.
        content_hash: SHA-256 hex digest of the content for integrity verification.
        config: Arbitrary configuration metadata associated with the prompt.
        required_variables: List of variable names that must be provided for rendering.
        label: Optional label (e.g. "production", "staging") if the prompt was
            resolved by label.
        cached_at: Optional ISO-8601 timestamp indicating when the response was cached.
    """

    schema_version: int
    name: str
    version: int
    content: str
    content_hash: str
    config: dict[str, Any]
    required_variables: tuple[str, ...]
    label: str | None = None
    cached_at: str | None = None

    def render(self, **variables: str) -> str:
        """Replace ``{{variable_name}}`` placeholders with provided values.

        This performs client-side rendering only (D21). The original ``content``
        field is never modified; a new string is returned.

        Required variables are taken from the server response, **not**
        re-extracted from the template.

        Args:
            **variables: Keyword arguments mapping variable names to their
                replacement values.

        Returns:
            A new string with all ``{{variable}}`` placeholders replaced.

        Raises:
            MissingVariableError: If any variable listed in
                ``required_variables`` is not supplied.
        """
        # Validate that all required variables are present
        for var_name in self.required_variables:
            if var_name not in variables:
                raise MissingVariableError(var_name, self.required_variables)

        # Warn about extra variables that are not in required_variables
        extra = set(variables.keys()) - set(self.required_variables)
        if extra:
            warnings.warn(
                f"Extra variables provided that are not in required_variables: "
                f"{sorted(extra)}. They will still be substituted if matching "
                f"placeholders exist in the template.",
                stacklevel=2,
            )

        def _replacer(match: re.Match[str]) -> str:
            var = match.group(1)
            if var in variables:
                return variables[var]
            return match.group(0)  # Leave unmatched placeholders intact

        return _VARIABLE_PATTERN.sub(_replacer, self.content)

    def verify_hash(self) -> bool:
        """Verify the content integrity against the stored SHA-256 hash.

        Returns:
            True if the SHA-256 of ``content`` matches ``content_hash``,
            False otherwise.
        """
        actual_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return actual_hash == self.content_hash
