"""Client-side PII/secret redaction for span input/output attributes.

Applied by ``@instrument`` when ``redact_by_default`` is enabled (WO-013 T1).
Matched values are replaced with a stable, non-reversible token
(``redacted:<sha256-prefix>``) so cardinality and joins survive while
plaintext does not. Only ``ev.data.inputs`` / ``ev.data.outputs`` are
scrubbed — correlation/interaction ids and ``ev.meta.*`` are never touched.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "api-key",
        "x-api-key",
        "password",
        "passwd",
        "pwd",
        "secret",
        "client_secret",
        "token",
        "access_token",
        "refresh_token",
        "session_token",
        "id_token",
        "bearer_token",
        "private_key",
        "credential",
        "credentials",
        "credit_card",
        "ssn",
    }
)

_VALUE_PATTERNS = (
    # JWT first: contains segments other patterns could partially match
    re.compile(r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),  # OpenAI/Stripe-style secret keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # 16-digit payment card
)

_KEYED_VALUE_PATTERN = re.compile(
    r"(?i)\b(authorization|api[_-]?key|apikey|password|passwd|secret|token)\b"
    r"(\s*[=:]\s*)([^\s\"'\\,;]+)"
)


def _token(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"redacted:{digest}"


def _replace_match(match: re.Match[str]) -> str:
    return _token(match.group(0))


def scrub_text(text: str) -> str:
    """Replace secret/PII-shaped substrings with stable redaction tokens."""
    for pattern in _VALUE_PATTERNS:
        text = pattern.sub(_replace_match, text)
    return _KEYED_VALUE_PATTERN.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{_token(m.group(3))}",
        text,
    )


def _is_sensitive_key(key: Any) -> bool:
    return str(key).lower() in _SENSITIVE_KEYS


def scrub_structured(value: Any) -> Any:
    """Recursively scrub a value, redacting sensitive keys and string patterns.

    Returns a new structure; the input is never mutated. Values under
    sensitive keys are replaced wholesale with a redaction token. Objects
    that are not plain containers/strings pass through unchanged — their
    serialized form is caught by :func:`scrub_text` afterwards.
    """
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {
            k: _token(str(v)) if _is_sensitive_key(k) else scrub_structured(v)
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return tuple(scrub_structured(v) for v in value)
    if isinstance(value, list):
        return [scrub_structured(v) for v in value]
    return value
