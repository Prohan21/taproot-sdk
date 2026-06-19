# AGENTS.md

This file provides guidance to Codex when working in `taproot-sdk/`. It is adapted from the local `CLAUDE.md`.

## Library Overview

`taproot-sdk` is the Python SDK for Taproot. It combines:
- OpenTelemetry-based instrumentation via `@instrument()` and provider auto-instrumentors
- `TaprootClient`, a unified async client across Taproot services
- The `taproot-tools` CLI for toolbox-oriented workflows

## Secret Handling Rule

Production runtime must never receive secret payloads or secret manager identifiers through environment variables. Services must derive canonical names like `taproot-<env>-<service>-<purpose>`, read secrets directly from the cloud secret manager once at startup using workload identity, and keep values in memory/settings/client objects. Do not write loaded secrets back to `os.environ`.

Forbidden in production runtime env:
- secret payloads: passwords, API keys, tokens, JWT secrets, provider keys
- secret identifiers: `*_SECRET_ARN`, `*_SECRET_URI`, `*_SECRET_RESOURCE`, `*_SECRET_NAME`
- platform injection: ECS `secrets`, Kubernetes `secretKeyRef`, Azure Container Apps `secret_name`, Cloud Run `secret_key_ref`

Only isolated, approval-gated bootstrap/rotation/operator jobs may handle secret identifiers or payloads.

## Commands

```bash
pip install -e ".[dev,all]"
pytest
pytest tests/test_decorators.py -v
pytest tests/test_decorators.py::test_name
mypy src/
ruff check src/
ruff format --check src/
pytest --cov=src/taproot_sdk --cov-report=term-missing
```

## Architecture

Key areas:
- `src/taproot_sdk/core.py` for SDK lifecycle and tracer setup
- `src/taproot_sdk/decorators.py` for `@instrument()`
- `src/taproot_sdk/auto_instrument.py` for dynamic provider instrumentation
- `src/taproot_sdk/client.py` for `TaprootClient`
- `src/taproot_sdk/otlp_json_exporter.py` for AWS-compatible OTLP JSON export
- `src/taproot_sdk/evals`, `guardrails`, `prompts`, `retrieval`, `toolbox`, and `workers` for typed models and service helpers

Important behaviors:
- Response models are frozen dataclasses and generally parse through `from_api_response()` or `from_dict()` helpers.
- `TaprootClient` supports APIM mode and direct mode with different routing and auth headers.
- Prompt fetching uses stale-while-revalidate caching.
- The OTLP JSON exporter exists specifically to avoid protobuf issues with AWS API Gateway v1.

## Editing Guidance

- Do not change package versioning unless explicitly asked.
- Preserve public SDK contracts, exception mappings, and immutable model semantics.
- Keep instrumentation provider-agnostic and avoid coupling decorators to specific LLM libraries.
- When changing client methods, consider APIM paths, direct-mode paths, and typed model parsing together.
