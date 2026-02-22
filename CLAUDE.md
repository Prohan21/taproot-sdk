# CLAUDE.md

OpenTelemetry-based instrumentation SDK for tracing LLM calls and custom functions. Exports spans via OTLP HTTP with gzip compression to Taproot observability platform.

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

- `src/taproot_sdk/__init__.py` — Public API exports: `init`, `shutdown`, `get_tracer`, `is_initialized`, `instrument`
- `src/taproot_sdk/core.py` — SDK initialization, TracerProvider setup, OTLP exporter configuration
- `src/taproot_sdk/decorators.py` — `@instrument()` decorator for custom function tracing
- `src/taproot_sdk/auto_instrument.py` — Auto-instrumentation loader for LLM libraries

### Key Exports

**Core functions** (`core.py`):
- `init()` — Initialize SDK with project_id, api_url, optional API key. Sets up TracerProvider with OTLP HTTP exporter, batch processor, and sampling. Returns Tracer instance. Registers atexit shutdown handler.
- `shutdown()` — Force flush pending spans and shutdown TracerProvider (10s timeout). Auto-called at exit.
- `get_tracer()` — Get configured Tracer instance (raises if not initialized)
- `is_initialized()` — Check SDK initialization state

**Decorator** (`decorators.py`):
- `@instrument(spankind, name, ignore_inputs, ignore_outputs, max_attribute_size)` — Decorator for tracing sync/async functions. Captures inputs/outputs as JSON, duration in ms, and exceptions. Supports partial input redaction via list of parameter names.

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
| `src/taproot_sdk/__init__.py` | Public API surface (5 exports) |
| `src/taproot_sdk/core.py` | SDK initialization, OTLP exporter setup |
| `src/taproot_sdk/decorators.py` | `@instrument()` decorator with sync/async support |
| `src/taproot_sdk/auto_instrument.py` | Dynamic LLM library instrumentation |
| `pyproject.toml` | Package metadata, dependencies, tool config (ruff line-length 100, mypy strict) |
| `tests/conftest.py` | Pytest fixtures with auto SDK reset |
