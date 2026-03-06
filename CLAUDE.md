# CLAUDE.md

This file provides guidance to Claude Code when working with the taproot-sdk codebase.

## Product Overview

The Taproot SDK (`taproot-sdk`) is a Python library that serves two purposes:

1. **OpenTelemetry-based instrumentation** for tracing LLM calls and custom functions, exporting spans via OTLP JSON with gzip compression to the Taproot observability platform (Evals-S).
2. **Unified async client** (`TaprootClient`) for interacting with all Taproot platform services (Retrieval-S, Evals-S, Guardrail-S, Prompt-S, ToolBox-S) through a single APIM gateway, with typed frozen-dataclass return models.

The SDK is published as `taproot-sdk` on PyPI. Version 0.1.0 (beta), Python 3.11+, Apache 2.0 license, built with Hatchling.

## Build & Development Commands

```bash
# Install for development (from taproot-sdk/)
pip install -e ".[dev,all]"
# Or with uv:
uv pip install -e ".[dev,all]"

# Run all tests
pytest                                     # or: uv run pytest
pytest tests/test_decorators.py -v         # single file
pytest tests/test_decorators.py::test_name # single test

# Type checking (strict mode)
mypy src/

# Linting (line-length 100)
ruff check src/
ruff format --check src/

# Coverage
pytest --cov=src/taproot_sdk --cov-report=term-missing
```

## Architecture

### Module Structure

```
src/taproot_sdk/
  __init__.py            Public API surface (all re-exports, ~100 symbols in __all__)
  core.py                SDK initialization, TracerProvider setup, OTLP exporter config
  decorators.py          @instrument() decorator for custom function tracing
  auto_instrument.py     Dynamic LLM library instrumentor loading (OpenLLMetry)
  client.py              TaprootClient — unified async HTTP client (~4200 lines)
  exceptions.py          Exception hierarchy (TaprootError -> TaprootAPIError -> specific)
  otlp_json_exporter.py  JsonOtlpSpanExporter — JSON OTLP for AWS API Gateway compat
  evals/
    __init__.py          Eval module exports
    models.py            EvalResult, RunHandle, GoldenDataset, TestConfiguration, etc.
    assertions.py        assert_eval() CI/CD assertion helper
    exceptions.py        EvalAssertionError
    pytest_plugin.py     eval_client, eval_run fixtures
  guardrails/
    __init__.py          Guardrails module exports
    models.py            GuardrailResponse, ScannerSignal, PolicySegmentResult, RedactionAction
    check_results.py     CheckResult, AnalyticsSummary, TimeseriesBucket
    configs.py           GuardrailConfig, ScannerOverride
  prompts/
    __init__.py          Prompts module exports
    models.py            PromptResponse, ChatMessage, ToolDefinition, PromptType
    client.py            PromptClient (DEPRECATED — use TaprootClient)
    cache.py             PromptCache — LRU with stale-while-revalidate
    exceptions.py        MissingVariableError
  retrieval/
    __init__.py          Retrieval module exports
    models.py            StoreInfo, QueryResponse, IngestionJob, BatchStatus, etc.
  toolbox/
    __init__.py          ToolBox module exports
    models.py            ToolInfo, InvocationResult, CredentialInfo, MCPServerInfo, etc.
    cli.py               taproot-tools CLI entry point (console_scripts)
tests/
  conftest.py            Autouse fixture: reset_sdk() calls shutdown() before/after each test
  test_core.py           TestInit, TestShutdown, TestGetTracer
  test_decorators.py     TestInstrumentDecorator sync/async/error handling
  test_auto_instrument.py Auto-instrumentation tests
  test_client.py         TaprootClient tests
  test_exceptions.py     Exception hierarchy tests
  auto_instrumentation/  Provider-specific tests (test_openai.py, test_anthropic.py)
  evals/                 test_models.py, test_assertions.py
  prompts/               test_client.py, test_models.py, test_cache.py, test_rendering.py, test_ab_testing.py
  retrieval/             test_models.py, test_client_methods.py
  toolbox/               test_models.py, test_client_methods.py, test_cli.py, test_tool_decorator.py, test_oauth_client.py, test_usage_report.py
```

### Key Design Patterns

- **Frozen dataclasses** for all response models (immutable, no mutation)
- **`from_api_response()` / `from_dict()` classmethods** on every model for JSON parsing
- **Tuple** (not list) for collection fields in frozen dataclasses
- **Async-first** client with sync convenience wrappers where needed
- **APIM vs direct mode** routing in TaprootClient (gateway vs local dev)
- **Stale-while-revalidate** cache for prompt serving
- **Typed exception hierarchy** wrapping HTTP errors with contextual info

## Core Functions (core.py)

### `init()`

Initialize the SDK with OpenTelemetry tracing. Must be called once at startup.

```python
import taproot_sdk as ev
ev.init(
    project_id="my-project",           # Required
    api_url="https://gateway.taproot.dev",  # Required — APIM gateway URL
    api_key="sk-...",                  # Optional — raw API key
    auto_instrument=["openai"],        # Optional — LLM libraries to auto-instrument
    redact_by_default=True,            # Optional (default True)
    sampling_rate=1.0,                 # Optional (default 1.0) — 0.0 to 1.0
    batch_size=512,                    # Optional — spans per export batch
    flush_interval_ms=5000,            # Optional — max time between flushes
    service_name="my-service",         # Optional — defaults to "taproot-{project_id}"
    service_version="1.0.0",           # Optional
)
```

**Internal details:**
- Creates `TracerProvider` with `Resource` containing `service.name` and `taproot.project_id`
- Sampler: `ParentBased(root=TraceIdRatioBased(sampling_rate))`
- Exporter: `JsonOtlpSpanExporter` (JSON, not protobuf — AWS API Gateway v1 corrupts binary)
- Endpoint: `{api_url}/api/v1/evals/v1/traces`
- Auth header: `x-api-key: {api_key}`
- Batch processor: `max_queue_size=batch_size*4`, `export_timeout_millis=30000`
- Registers `atexit` shutdown handler
- Raises `RuntimeError` if called twice without `shutdown()` first

### `shutdown()`

Force flush pending spans and shutdown TracerProvider (10s timeout). Auto-called at exit. Resets all global state.

### `get_tracer()` / `is_initialized()` / `get_config()`

Accessors for the global tracer, initialization state, and config dict (copy).

## @instrument() Decorator (decorators.py)

Wraps sync or async functions to automatically create OpenTelemetry spans.

```python
@ev.instrument(
    spankind="tool",                   # Span type (see 9 kinds below)
    name="custom-span-name",           # Override function name
    ignore_inputs=False,               # True = skip all; list[str] = skip named params
    ignore_outputs=False,              # True = skip return value capture
    max_attribute_size=65536,          # 64KB default, truncates with ...[TRUNCATED]
)
def my_function(query: str, api_key: str) -> dict:
    ...
```

### 9 Span Kinds

| Span Kind | Description |
|-----------|-------------|
| `workflow` | End-to-end pipeline (default) |
| `agent` | Autonomous agent operations |
| `chain` | Sequential processing |
| `tool` | External tool/function calls |
| `retrieval` | Knowledge base/RAG operations |
| `embedding` | Vector embedding generation |
| `completion` | Text generation (non-chat) |
| `chat` | Conversational LLM calls |
| `rerank` | Result reordering |

### Span Attributes Set

| Attribute | Description |
|-----------|-------------|
| `ev.type.node` | Span kind string |
| `ev.meta.function` | Function name |
| `ev.meta.module` | Module name |
| `ev.data.inputs` | JSON-serialized inputs |
| `ev.data.outputs` | JSON-serialized outputs |
| `ev.metrics.duration_ms` | Execution time in milliseconds |

**Truncation:** When attribute exceeds `max_attribute_size`, truncates with `...[TRUNCATED]` suffix and adds `{key}_size` (int) and `{key}_truncated` (bool) attributes.

**Serialization:** Uses custom `_json_default` that handles Pydantic v1/v2 models, objects with `__dict__`, numpy arrays (`.tolist()`), and falls back to `repr()`.

**Graceful degradation:** Works without `ev.init()` — uses OpenTelemetry's noop tracer (no spans exported, no errors).

## Auto-Instrumentation (auto_instrument.py)

Dynamic loading of OpenTelemetry instrumentors for 7 LLM providers via OpenLLMetry packages.

| Library Key | Instrumentor Class | Install Extra |
|-------------|-------------------|---------------|
| `openai` | `OpenAIInstrumentor` | `taproot-sdk[openai]` |
| `anthropic` | `AnthropicInstrumentor` | `taproot-sdk[anthropic]` |
| `google` | `GoogleGenerativeAiInstrumentor` | `taproot-sdk[google]` |
| `cohere` | `CohereInstrumentor` | `taproot-sdk[cohere]` |
| `vertexai` | `VertexAIInstrumentor` | `taproot-sdk[vertexai]` |
| `bedrock` | `BedrockInstrumentor` | `taproot-sdk[bedrock]` |
| `mistral` | `MistralAiInstrumentor` | `taproot-sdk[mistral]` |

**How it works:**
- `setup_auto_instrumentation(libraries)` — loads instrumentors via `importlib`, calls `.instrument()`, tracks in `_initialized_instrumentors` set to prevent double-instrumentation
- `uninstrument(libraries)` / `uninstrument_all()` — calls `.uninstrument()` and removes from tracking set
- `get_instrumented_libraries()` / `is_instrumented(library)` — query state
- Missing optional deps logged at DEBUG level (expected if user hasn't installed the extra)

## OTLP JSON Exporter (otlp_json_exporter.py)

`JsonOtlpSpanExporter` — custom `SpanExporter` that serializes spans as OTLP JSON instead of protobuf.

**Why:** AWS API Gateway REST API v1 does not support binary media types by default and corrupts protobuf payloads.

**How:**
1. Uses `encode_spans()` from OTel protobuf encoder to build canonical `ExportTraceServiceRequest`
2. Converts to dict via `google.protobuf.json_format.MessageToDict`
3. Normalizes trace/span/parent IDs from base64 to lowercase hex (OTLP JSON spec)
4. Serializes with `json.dumps(separators=(",", ":"))` for compactness
5. Optionally gzip-compresses the body (enabled by default)
6. Sends via `httpx.post()` with `Content-Type: application/json` and `Content-Encoding: gzip`

## TaprootClient (client.py)

Single async HTTP client for all Taproot platform services. ~4200 lines, ~120 async methods.

### Constructor

```python
client = TaprootClient(
    base_url="https://gateway.taproot.dev",  # APIM gateway URL (required)
    api_key="your-raw-api-key",               # API key (required)
    project_id="my-project",                  # Default project ID
    timeout=30.0,                             # HTTP timeout in seconds
    direct_mode=False,                        # False=APIM, True=direct service
    cache_ttl_seconds=30.0,                   # Prompt cache TTL
    max_stale_seconds=60.0,                   # Prompt cache max staleness
)
```

Falls back to `ev.init()` config for `base_url`, `api_key`, `project_id` if not explicitly provided.

### APIM vs Direct Mode

| Mode | Auth Header | Path Prefix Example |
|------|-------------|-------------------|
| APIM (default) | `x-api-key: {raw_key}` | `/api/v1/evals/v1/projects/{pid}/...` |
| Direct | `X-Api-Key-Id: {key_id}` | `/v1/projects/{pid}/...` |

### Retry Logic

- Retries on status codes: 429, 500, 502, 503, 504 and connection errors
- Exponential backoff: 1s, 2s, 4s, 8s, 10s (capped at `_MAX_RETRY_WAIT=10`)
- Max 5 retries (`_MAX_RETRIES=5`)
- Total attempts and wait time tracked and surfaced in `ServerError`

### Error Handling

All non-2xx responses are mapped to typed exceptions:

| Status | Exception | Notes |
|--------|-----------|-------|
| 401/403 | `AuthenticationError` | Includes project_id context, actionable hints |
| 404 | `TaprootAPIError` or `PromptNotFoundError` | Prompt-specific subclass for prompt endpoints |
| 409 | `ConflictError` | Duplicate resources |
| 422 | `ValidationError` | Includes `.errors` list from FastAPI/Guardrail-S formats |
| 429 | `RateLimitError` | Includes `.retry_after` from header |
| 5xx | `ServerError` | Includes `.attempts` and `.total_wait_seconds` |

### Service Method Groups

**Retrieval-S** (Stores, Access, Query, Ingestion, Batch, Documents, Chunks):
- `create_store()`, `list_stores()`, `get_store()`, `get_store_by_name()`, `update_store()`, `delete_store()`, `get_store_stats()`
- `grant_store_access()`, `revoke_store_access()`, `list_store_access()`
- `retrieval_query()` — vector/hybrid/keyword search with reranking
- `ingest_document()`, `get_ingestion_job()`, `list_ingestion_jobs()`, `cancel_ingestion_job()`
- `batch_ingest()`, `get_batch_status()`, `list_batch_jobs()`, `cancel_batch()`
- `list_documents()`, `get_document()`, `delete_document()`
- `upload_chunks()`, `list_chunks()`, `get_chunk()`, `delete_chunks()`

**Prompt-S Serving** (cached reads):
- `get_prompt(name, version, label)` — L1 cached with stale-while-revalidate
- `get_prompt_sync()` — sync convenience wrapper
- `prompt_cache` property for manual invalidation

**Prompt-S Management** (CRUD, versioning, labels, approval, examples, optimization):
- `create_prompt()`, `list_prompts()`, `get_prompt_metadata()`, `update_prompt()`, `archive_prompt()`, `delete_prompt()`
- `create_version()`, `list_versions()`, `get_version()`, `diff_versions()`
- `set_label()`, `set_weighted_label()`, `list_labels()`, `get_label()`, `delete_label()`
- `approve_version()`, `reject_version()` — four-eyes rule
- `add_examples()`, `list_examples()`
- `trigger_optimization()`, `list_optimization_jobs()`

**Guardrail-S** (content scanning, configs, analytics):
- `check_input(content)` -> `GuardrailResponse`
- `check_output(content, original_input)` -> `GuardrailResponse`
- `list_check_results()`, `get_check_result()`, `delete_check_result()`
- `get_guardrail_analytics(start_date, end_date, granularity)`
- `create_guardrail_config()`, `list_guardrail_configs()`, `get_guardrail_config()`, `update_guardrail_config()`, `delete_guardrail_config()`, `set_default_guardrail_config()`

**Evals-S** (datasets, configs, runs, experiments, alerts, webhooks, discovery, traces):
- Golden Datasets: `create_dataset()`, `list_datasets()`, `get_dataset()`, `update_dataset()`, `delete_dataset()`, `add_dataset_item()`, `list_dataset_items()`, `update_dataset_item()`, `delete_dataset_item()`, `bulk_session_assign()`, `promote_trace_to_dataset()`, `create_dataset_version()`, `list_dataset_versions()`
- Test Configs: `create_test_config()`, `list_test_configs()`, `get_test_config()`, `update_test_config()`, `delete_test_config()`
- Test Runs: `trigger_eval_run()`, `list_eval_runs()`, `get_eval_run()`, `cancel_eval_run()`, `get_eval_run_results()`, `compare_eval_runs()`, `wait_for_eval()`
- Experiments: `create_experiment()`, `list_experiments()`, `get_experiment()`, `update_experiment()`, `delete_experiment()`, `compare_experiment_runs()`
- Alerts: `create_alert_rule()`, `list_alert_rules()`, `get_alert_rule()`, `update_alert_rule()`, `delete_alert_rule()`, `get_alert_history()`
- Webhooks: `create_webhook()`, `list_webhooks()`, `get_webhook()`, `update_webhook()`, `delete_webhook()`, `list_webhook_deliveries()`, `get_webhook_delivery()`, `retry_webhook_delivery()`
- Discovery: `start_discovery()`, `list_discovery_sessions()`, `get_discovery_session()`, `get_discovery_probes()`, `get_discovery_suggestions()`, `update_discovery_suggestion()`, `approve_discovery_suggestions()`, `reject_discovery_suggestion()`, `cancel_discovery()`
- Traces: `ingest_traces()`, `list_traces()`, `get_trace()`, `get_trace_stats()`, `delete_trace()`
- Export: `export_eval_results()`, `get_eval_job_status()`

**ToolBox-S** (tool management, credentials, MCP, OpenAPI import):
- `push_tool()`, `register_tool()`, `invoke_tool()`, `list_tools()`, `discover_tools()`, `get_tool()`, `update_tool()`, `delete_tool()`
- `import_openapi()`, `import_mcp_registry()`, `export_mcp_registry()`
- `set_tool_credential()`, `list_credentials()`, `revoke_credential()`
- `start_oauth_flow()`, `complete_oauth_flow()`, `client_credentials_grant()`
- `register_mcp_server()`, `list_mcp_servers()`, `get_mcp_server()`, `delete_mcp_server()`
- `get_usage_report()`, `push_decorated_tools()`

**Health checks:** `health_retrieval()`, `health_evals()`, `health_evals_basic()`, `health_guardrails()`, `health_prompts()`, `health_toolbox()`

## Guardrails Client

### check_input() / check_output()

```python
response = await client.check_input(
    "user message content",
    project_id="my-project",       # Optional — uses client default
    config_name="strict",          # Optional — named config
    trace_id="abc-123",            # Optional — auto-generates UUID v4 if omitted
    scanner_overrides={"toxicity": False},  # Check-level: Dict[str, bool]
)

response = await client.check_output(
    "LLM response content",
    original_input="the user's original question",  # Optional — for context
    scanner_overrides={"prompt_injection": True},
)
```

### GuardrailResponse Model

```python
response.verdict        # "ALLOW", "BLOCK", "ALLOW_WITH_REDACTIONS"
response.is_blocked     # True if verdict == "BLOCK"
response.is_allowed     # True if verdict == "ALLOW"
response.signals        # Tuple[ScannerSignal, ...] — individual scanner results
response.flagged_scanners  # Property — scanners that flagged content
response.policy_signals    # Tuple[PolicySegmentResult, ...] — LLM judge results
response.violated_policies # Property — violated policy segments
response.redactions     # Tuple[RedactionAction, ...] | None
response.blocked_by     # str | None — which scanner caused the block
response.block_reasons  # Tuple[str, ...] | None
response.total_latency_ms  # float
response.request_id     # str
```

### ScannerSignal Model

```python
signal.scanner_id       # "toxicity", "prompt_injection", etc.
signal.scanner_version  # str
signal.is_valid         # True if content passed this scanner
signal.flagged          # Property: not is_valid
signal.score            # float | None
signal.labels           # Tuple[str, ...]
signal.reasoning        # str | None (LLM judge)
signal.latency_ms       # float
```

### Scanner Overrides: Two Schemas

- **Check-level** (in `check_input`/`check_output`): `Dict[str, bool]` — enable/disable scanners per request
- **Config-level** (in `create_guardrail_config`): `Dict[str, {"enabled": bool, "threshold": float, "config": dict}]` — full scanner config via `ScannerOverride` model

## Prompt Client

### Via TaprootClient (recommended)

```python
prompt = await client.get_prompt("welcome-email", label="production")
rendered = prompt.render(user_name="Alice", plan="Pro")

# Chat prompts
chat_prompt = await client.get_prompt("chat-template")
messages = chat_prompt.render_messages(topic="weather")

# LLM provider convenience
openai_messages = prompt.to_openai_messages(user_name="Alice")
system, messages = prompt.to_anthropic_messages(user_name="Alice")
```

### PromptResponse Model

```python
prompt.schema_version    # int (currently 1)
prompt.name              # str
prompt.version           # int — resolved version number
prompt.content           # str — raw template with {{variable}} placeholders
prompt.content_hash      # str — SHA-256 hex digest
prompt.config            # dict — arbitrary metadata
prompt.required_variables # Tuple[str, ...]
prompt.label             # str | None
prompt.prompt_type       # PromptType.TEXT or PromptType.CHAT
prompt.messages          # Tuple[ChatMessage, ...] | None (CHAT only)
prompt.tools             # Tuple[ToolDefinition, ...] | None
prompt.ab_test           # bool — True if served via weighted label
prompt.selected_variant  # int | None — which variant was selected
```

### Rendering (Client-Side Only)

Variable rendering is **client-side only** (decision D21). The server stores and serves templates with `{{variables}}` intact. Values are never sent to the server.

- `render(**variables)` — for TEXT prompts, returns string with placeholders replaced
- `render_messages(**variables)` — for CHAT prompts, returns new tuple of ChatMessage
- Raises `MissingVariableError` if any `required_variables` are not provided
- Warns about extra variables not in `required_variables`
- `verify_hash()` — SHA-256 integrity check of content against content_hash (constant-time via `hmac.compare_digest`)
- `to_openai_messages()` / `to_anthropic_messages()` — provider-specific formatting
- `to_dict()` — serialize to plain dict

### PromptClient (DEPRECATED)

`PromptClient` in `prompts/client.py` is deprecated. Use `TaprootClient` instead. It issues a `DeprecationWarning` on construction. It talks directly to the Prompt-S serving layer (Lambda/Azure Functions) at `/serve/{project_id}/{name}`.

### Prompt Cache (prompts/cache.py)

`PromptCache` — in-memory TTL cache with stale-while-revalidate semantics.

- **Fresh** (age < `ttl_seconds`): Return immediately
- **Stale** (age < `ttl_seconds + max_stale_seconds`): Return stale, schedule background revalidation
- **Expired** (age >= `ttl_seconds + max_stale_seconds`): Block on fresh fetch
- Cache key: `"{project_id}:{name}:v={version}:l={label}"`
- Thread-safe: `asyncio.Lock` for async, `threading.Lock` for sync
- Background revalidation via `asyncio.ensure_future()` (async only)
- `invalidate()` / `clear()` for manual cache management

## Evals Integration

### CI/CD Workflow

```python
# Trigger and wait
handle = await client.trigger_eval_run(test_config_id, tags=["ci"])
result = await client.wait_for_eval(handle.run_id, timeout=300, poll_interval=5)

# Assert quality gates
from taproot_sdk import assert_eval
assert_eval(result, min_pass_rate=80, min_score={"answer_relevancy": 0.7}, max_duration_ms=60000)
```

### EvalResult Model

```python
result.run_id           # str
result.status           # "completed", "failed", "running", etc.
result.total_items      # int
result.completed_items  # int
result.failed_items     # int
result.pass_rate        # Property: percentage (0-100)
result.duration_ms      # Property: float | None
result.aggregate_scores # Dict[str, AggregateScore] — {metric: {mean, min, max, std_dev, passed, failed}}
result.tags             # list[str]
result.error_message    # str | None
```

### Pytest Plugin

Register in conftest.py:
```python
pytest_plugins = ["taproot_sdk.evals.pytest_plugin"]
```

Provides `eval_client` fixture (from env vars: `TAPROOT_EVAL_URL`, `TAPROOT_API_KEY_ID`, `TAPROOT_PROJECT_ID`) and `eval_run` factory fixture.

## ToolBox-S CLI

The SDK provides a `taproot-tools` CLI command (registered via `[project.scripts]`).

```bash
# Environment variables (required)
export TAPROOT_BASE_URL="https://gateway.taproot.dev"
export TAPROOT_API_KEY="your-api-key"
export TAPROOT_PROJECT_ID="my-project"
export TAPROOT_DIRECT_MODE="false"  # optional

# Commands
taproot-tools push <file> --name <name> --entry-point <func>
taproot-tools register <name> --endpoint-url <url>
taproot-tools invoke <name> --input '{"key": "value"}'
taproot-tools search "find weather tools"
taproot-tools list [--tags <tags>] [--type hosted|external|mcp]
taproot-tools get <tool-id>
taproot-tools delete <tool-id>
taproot-tools import-openapi --namespace <ns> --spec-url <url>
taproot-tools import-mcp-registry --registry-url <url>
taproot-tools export-mcp-registry [--output <file>]
taproot-tools set-credential --tool-id <id> --type api_key --name <name> --payload '{...}'
taproot-tools list-credentials [--tool-id <id>]
taproot-tools revoke-credential --credential-id <id> --version <v>
taproot-tools mcp-register --name <name> --transport sse --url <url>
taproot-tools mcp-list
taproot-tools mcp-get --id <id>
taproot-tools mcp-delete --id <id>

# All commands support --json for machine-readable output
```

## Exception Hierarchy

```
TaprootError (base)
  TaprootAPIError (HTTP errors — has status_code, detail, service, request_url, body)
    PromptNotFoundError (404 for prompts — has prompt_name, prompt_project_id, prompt_version, prompt_label)
    AuthenticationError (401/403 — has auth_project_id, actionable hints)
    ConflictError (409)
    RateLimitError (429 — has retry_after)
    ServerError (5xx — has attempts, total_wait_seconds)
    ValidationError (422 — has errors list)
  MissingVariableError (prompt rendering — has variable_name, required_variables)

EvalAssertionError (AssertionError subclass — has result attribute)
```

## Configuration

### Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `TAPROOT_EVAL_URL` | pytest_plugin | Base URL for eval_client fixture |
| `TAPROOT_API_KEY_ID` | pytest_plugin | API key for eval_client fixture |
| `TAPROOT_PROJECT_ID` | pytest_plugin | Project ID for eval_client fixture |
| `TAPROOT_BASE_URL` | CLI | API gateway URL |
| `TAPROOT_API_KEY` | CLI | API key |
| `TAPROOT_DIRECT_MODE` | CLI | "true" for direct mode |

### Programmatic Configuration

Configuration is passed to `ev.init()` for tracing, or to `TaprootClient()` constructor for the HTTP client. If `ev.init()` was called, `TaprootClient()` pulls from the SDK config for any parameters not explicitly provided.

## Installation

```bash
# Core SDK (tracing + client)
pip install taproot-sdk

# With specific LLM auto-instrumentation
pip install taproot-sdk[openai]
pip install taproot-sdk[anthropic]
pip install taproot-sdk[google]
pip install taproot-sdk[cohere]
pip install taproot-sdk[bedrock]
pip install taproot-sdk[vertexai]
pip install taproot-sdk[mistral]

# All LLM providers
pip install taproot-sdk[all]

# Development
pip install taproot-sdk[dev,all]
```

### Core Dependencies

- `opentelemetry-api>=1.21.0`
- `opentelemetry-sdk>=1.21.0`
- `opentelemetry-exporter-otlp-proto-http>=1.21.0`
- `httpx>=0.25.0`

### Dev Dependencies

- `pytest>=7.0.0`, `pytest-asyncio>=0.21.0`, `pytest-cov>=4.0.0`
- `mypy>=1.0.0` (strict mode)
- `ruff>=0.1.0` (line-length 100)
- `respx>=0.20.0` (HTTP mocking for tests)

## Testing Conventions

- **pytest** with `asyncio_mode = "auto"` (pyproject.toml)
- **Autouse fixture** `reset_sdk()` in `tests/conftest.py` calls `shutdown()` before and after each test to ensure clean global state
- All response models have `from_api_response()` / `from_dict()` tests
- Client methods tested with `respx` for HTTP mocking
- Decorator tests verify both sync and async functions, error recording, input/output capture, and truncation
- No live network calls in unit tests

Run tests:
```bash
pytest                           # All tests
pytest tests/ -v --tb=short      # Verbose with short traceback
pytest --cov=src/taproot_sdk     # With coverage
```

## Integration Patterns

### End-User Application

```python
import taproot_sdk as ev

# 1. Initialize tracing
ev.init(project_id="my-app", api_url="https://gw.taproot.dev", api_key="sk-...")

# 2. Auto-trace LLM calls
import openai
client = openai.OpenAI()  # Automatically instrumented

# 3. Trace custom functions
@ev.instrument(spankind="retrieval")
def search(query: str) -> list:
    ...

# 4. Use platform services
async with ev.TaprootClient() as tc:
    prompt = await tc.get_prompt("welcome")
    guard = await tc.check_input(user_input)
    if guard.is_blocked:
        return "Content blocked"
    results = await tc.retrieval_query("my-store", user_input)
```

### CI/CD Pipeline

```python
import taproot_sdk as ev

async def test_agent_quality():
    async with ev.TaprootClient(
        base_url="https://gw.taproot.dev",
        api_key="ci-key",
        project_id="my-project",
    ) as client:
        handle = await client.trigger_eval_run("test-config-id")
        result = await client.wait_for_eval(handle.run_id, timeout=300)
        ev.assert_eval(result, min_pass_rate=80)
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/taproot_sdk/__init__.py` | 188 | Public API surface (~100 exports) |
| `src/taproot_sdk/core.py` | 251 | SDK init, TracerProvider, OTLP exporter setup |
| `src/taproot_sdk/decorators.py` | 294 | @instrument() decorator with 9 span kinds |
| `src/taproot_sdk/auto_instrument.py` | 200 | Dynamic LLM instrumentor loading |
| `src/taproot_sdk/client.py` | ~4200 | TaprootClient — ~120 async methods for all services |
| `src/taproot_sdk/exceptions.py` | 240 | Typed exception hierarchy |
| `src/taproot_sdk/otlp_json_exporter.py` | 147 | JSON OTLP exporter for API Gateway compat |
| `src/taproot_sdk/evals/models.py` | 722 | 15+ eval-related frozen dataclasses |
| `src/taproot_sdk/guardrails/models.py` | 152 | GuardrailResponse, ScannerSignal, etc. |
| `src/taproot_sdk/prompts/models.py` | 319 | PromptResponse with render/verify |
| `src/taproot_sdk/prompts/cache.py` | 269 | Stale-while-revalidate prompt cache |
| `src/taproot_sdk/retrieval/models.py` | 758 | 20+ retrieval frozen dataclasses |
| `src/taproot_sdk/toolbox/models.py` | 391 | ToolInfo, InvocationResult, MCPServerInfo, etc. |
| `src/taproot_sdk/toolbox/cli.py` | 866 | taproot-tools CLI with 16 subcommands |
