# CLAUDE.md

OpenTelemetry-based instrumentation SDK for tracing LLM calls and custom functions. Exports spans via OTLP HTTP with gzip compression to Taproot observability platform. Also provides a unified `TaprootClient` for interacting with all Taproot platform services (Evals-S, Prompt-S, Retrieval-S) behind the APIM gateway, with typed return models for evals and prompts.

## Build & Install Commands

```bash
# Install for development (from repo root)
cd taproot-sdk
pip install -e ".[dev,all]"

# Install from PyPI (production)
pip install taproot-sdk              # Core SDK
pip install taproot-sdk[openai]      # With OpenAI instrumentation
pip install taproot-sdk[all]         # All LLM provider instrumentations

# Run tests
pytest                                  # All tests
pytest tests/test_decorators.py -v     # Single test file
pytest -v --tb=short                   # Verbose with short traceback

# Type checking
mypy src/

# Linting
ruff check src/
ruff format --check src/
```

## Package Metadata

- **PyPI name**: `taproot-sdk`
- **Python**: 3.9+
- **Version**: 0.1.0 (beta)
- **License**: Apache 2.0
- **Build system**: Hatchling

## Architecture

### Module Structure

- `src/taproot_sdk/__init__.py` — Public API surface: `init`, `shutdown`, `get_tracer`, `is_initialized`, `instrument`, `TaprootClient`, `EvalResult`, `RunHandle`, `assert_eval`, `EvalAssertionError`, `PromptClient`, `PromptResponse`, `MissingVariableError`
- `src/taproot_sdk/core.py` — SDK initialization, TracerProvider setup, OTLP exporter configuration
- `src/taproot_sdk/decorators.py` — `@instrument()` decorator for custom function tracing
- `src/taproot_sdk/auto_instrument.py` — Auto-instrumentation loader for LLM libraries
- `src/taproot_sdk/client.py` — Unified async HTTP client for all Taproot platform services via APIM gateway
- `src/taproot_sdk/evals/` — Eval models, assertions, and pytest plugin
- `src/taproot_sdk/prompts/` — Prompt client with caching, rendering, and integrity verification

### Key Exports

**Core functions** (`core.py`):
- `init()` — Initialize SDK with project_id, api_url, optional API key. Sets up TracerProvider with OTLP HTTP exporter, batch processor, and sampling. Returns Tracer instance. Registers atexit shutdown handler.
- `shutdown()` — Force flush pending spans and shutdown TracerProvider (10s timeout). Auto-called at exit.
- `get_tracer()` — Get configured Tracer instance (raises if not initialized)
- `is_initialized()` — Check SDK initialization state

**Decorator** (`decorators.py`):
- `@instrument(spankind, name, ignore_inputs, ignore_outputs, max_attribute_size)` — Decorator for tracing sync/async functions. Captures inputs/outputs as JSON, duration in ms, and exceptions. Supports partial input redaction via list of parameter names.

**Unified Client** (`client.py` — `TaprootClient`):
- Constructor: `base_url, api_key, project_id, timeout`. Pulls from SDK config if `ev.init()` was called.
- All methods are async and return typed models (not raw dicts).
- Evals: `trigger_eval_run() -> RunHandle`, `get_eval_run() -> EvalResult`, `wait_for_eval() -> EvalResult`
- Prompts: `get_prompt() -> PromptResponse` (renders via `response.render(var=val)`)
- Retrieval: `retrieval_query() -> dict` (raw dict — no typed model yet)
- Health: `health_retrieval()`, `health_evals()`, `health_guardrails()`

**Evals** (`evals/`):
- `EvalResult` — Frozen dataclass: `run_id`, `status`, `total_items`, `completed_items`, `failed_items`, `aggregate_scores`, `started_at`, `completed_at`, `error_message`, `tags`. Properties: `pass_rate`, `duration_ms`. Classmethod: `from_api_response(data)`.
- `RunHandle` — Frozen dataclass: `run_id`, `status`, `message`
- `AggregateScore` — Frozen dataclass: `mean`, `min`, `max`, `std_dev`, `passed`, `failed`
- `assert_eval(result, *, min_pass_rate, min_score, max_duration_ms)` — CI/CD assertion helper. Raises `EvalAssertionError` on failure.
- `EvalAssertionError` — Exception subclass with `.result` attribute for debugging.
- `pytest_plugin.py` — Provides `eval_client` and `eval_run` fixtures. Reads `TAPROOT_EVAL_URL`, `TAPROOT_API_KEY_ID`, `TAPROOT_PROJECT_ID` from env.

**Prompts** (`prompts/`):
- `PromptClient` — Standalone async client with L1 in-memory cache (stale-while-revalidate). Has `get()` async and `get_sync()` convenience wrapper.
- `PromptResponse` — Frozen dataclass: `schema_version`, `name`, `version`, `content`, `content_hash`, `config`, `required_variables`, `label`, `cached_at`. Methods: `render(**variables)`, `verify_hash()`.
- `MissingVariableError` — Raised when `render()` is called without required variables.

### Span Kinds (9 types)

Defined in `decorators.py` as Literal type:
- `workflow` — End-to-end pipeline (default)
- `agent` — Autonomous agent operations
- `chain` — Sequential processing
- `tool` — External tool/function calls
- `retrieval` — Knowledge base/RAG operations
- `embedding` — Vector embedding generation
- `completion` — Text generation (non-chat)
- `chat` — Conversational LLM calls
- `rerank` — Result reordering

### Auto-Instrumentation

`auto_instrument.py` provides dynamic loading of OpenTelemetry instrumentors for LLM libraries. Mapping defined in `INSTRUMENTORS` dict:
- `openai` → `opentelemetry.instrumentation.openai.OpenAIInstrumentor`
- `anthropic` → `opentelemetry.instrumentation.anthropic.AnthropicInstrumentor`
- `google` → `opentelemetry.instrumentation.google_generativeai.GoogleGenerativeAiInstrumentor`
- `cohere` → `opentelemetry.instrumentation.cohere.CohereInstrumentor`
- `vertexai` → `opentelemetry.instrumentation.vertexai.VertexAIInstrumentor`
- `bedrock` → `opentelemetry.instrumentation.bedrock.BedrockInstrumentor`
- `mistral` → `opentelemetry.instrumentation.mistralai.MistralAiInstrumentor`

Instrumentors are lazily loaded via `_load_instrumentor()` using `importlib`. Tracks initialized libraries in `_initialized_instrumentors` set to prevent double-instrumentation.

### Span Attributes

Set by `@instrument()` decorator:
- `ev.type.node` — Span kind (workflow, tool, etc.)
- `ev.meta.function` — Function name
- `ev.meta.module` — Module name
- `ev.data.inputs` — JSON-serialized inputs (respects ignore_inputs)
- `ev.data.outputs` — JSON-serialized outputs (respects ignore_outputs)
- `ev.metrics.duration_ms` — Execution time in milliseconds

Truncation handling: If attribute exceeds `max_attribute_size` (default 64KB), truncates with "...[TRUNCATED]" suffix and sets `{key}_size` and `{key}_truncated` attributes.

## Configuration

`init()` parameters (all verified in `core.py`):
- `project_id` (required) — Project identifier, added to resource as `taproot.project_id`
- `api_url` (required) — Taproot backend URL (trailing slash stripped), appends `/v1/traces` for OTLP endpoint
- `api_key` (optional) — Auth token, sent as `Authorization: Bearer {api_key}` header
- `auto_instrument` (optional) — List of library names to auto-instrument on init
- `redact_by_default` (bool, default True) — Stored in config (implementation TBD)
- `sampling_rate` (float, default 1.0) — Trace sampling rate (0.0-1.0), uses `ParentBased(TraceIdRatioBased)` sampler
- `batch_size` (int, default 512) — Max spans per export batch
- `flush_interval_ms` (int, default 5000) — Max time between batch exports
- `service_name` (optional) — Service name resource attribute (defaults to `taproot-{project_id}`)
- `service_version` (optional) — Service version resource attribute

Batch processor config (verified in `core.py`):
- `max_queue_size` = batch_size * 4
- `max_export_batch_size` = batch_size
- `schedule_delay_millis` = flush_interval_ms
- `export_timeout_millis` = 30000 (30s)

## Testing Conventions

- pytest with `asyncio_mode = "auto"` (pyproject.toml)
- `tests/conftest.py` — Autouse fixture `reset_sdk()` calls `shutdown()` before/after each test
- Test classes: `TestInit`, `TestShutdown`, `TestGetTracer`, `TestInstrumentDecorator`, `TestInstrumentWithoutInit`
- Decorator gracefully handles uninitialized SDK (uses OpenTelemetry noop tracer)
- No mocks or network interception (tests run against real OTLP exporter with localhost endpoint)

## Dependencies

**Core** (pyproject.toml):
- `opentelemetry-api>=1.21.0`
- `opentelemetry-sdk>=1.21.0`
- `opentelemetry-exporter-otlp-proto-http>=1.21.0`

**Optional** (LLM provider instrumentations):
- Each LLM library has dedicated `opentelemetry-instrumentation-{provider}>=0.27.0` package
- Install via extras: `[openai]`, `[anthropic]`, `[google]`, `[cohere]`, `[bedrock]`, `[vertexai]`, `[mistral]`, or `[all]`

**Dev** (pytest, mypy, ruff, respx, httpx)

## Key Files

| File | Purpose |
|------|---------|
| `src/taproot_sdk/__init__.py` | Public API surface (13 exports) |
| `src/taproot_sdk/core.py` | SDK initialization, OTLP exporter setup |
| `src/taproot_sdk/decorators.py` | `@instrument()` decorator with sync/async support |
| `src/taproot_sdk/auto_instrument.py` | Dynamic LLM library instrumentation |
| `src/taproot_sdk/client.py` | `TaprootClient` — unified async client for all services |
| `src/taproot_sdk/evals/__init__.py` | Eval module exports |
| `src/taproot_sdk/evals/models.py` | `EvalResult`, `RunHandle`, `AggregateScore` frozen dataclasses |
| `src/taproot_sdk/evals/assertions.py` | `assert_eval()` CI/CD assertion helper |
| `src/taproot_sdk/evals/exceptions.py` | `EvalAssertionError` with `.result` attribute |
| `src/taproot_sdk/evals/pytest_plugin.py` | `eval_client` and `eval_run` pytest fixtures |
| `src/taproot_sdk/prompts/client.py` | `PromptClient` — standalone prompt client with caching |
| `src/taproot_sdk/prompts/models.py` | `PromptResponse` frozen dataclass with `render()` and `verify_hash()` |
| `src/taproot_sdk/prompts/cache.py` | `PromptCache` — LRU with stale-while-revalidate |
| `src/taproot_sdk/prompts/exceptions.py` | `MissingVariableError` |
| `pyproject.toml` | Package metadata, dependencies, tool config (ruff line-length 100, mypy strict) |
| `tests/conftest.py` | Pytest fixtures with auto SDK reset |
| `tests/evals/test_models.py` | Tests for `EvalResult`, `RunHandle` |
| `tests/evals/test_assertions.py` | Tests for `assert_eval()` with pass/fail scenarios |
