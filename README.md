# Taproot SDK

Developer SDK for integrating applications with Taproot. It provides OpenTelemetry-based instrumentation, typed service clients, and correlation metadata needed for traces, evaluations, diagnosis, and System of Record linkage.

The SDK is the adoption edge: it should make Taproot easy to add to customer code without turning SDK-provided metadata into authorization or audit authority.

## Secret Handling Rule

Production runtime must never receive secret payloads or secret manager identifiers through environment variables. Services must derive canonical names like `taproot-<env>-<service>-<purpose>`, read secrets directly from the cloud secret manager once at startup using workload identity, and keep values in memory/settings/client objects. Do not write loaded secrets back to `os.environ`.

Forbidden in production runtime env:
- secret payloads: passwords, API keys, tokens, JWT secrets, provider keys
- secret identifiers: `*_SECRET_ARN`, `*_SECRET_URI`, `*_SECRET_RESOURCE`, `*_SECRET_NAME`
- platform injection: ECS `secrets`, Kubernetes `secretKeyRef`, Azure Container Apps `secret_name`, Cloud Run `secret_key_ref`

Only isolated, approval-gated bootstrap/rotation/operator jobs may handle secret identifiers or payloads.

## Features

- **Automatic LLM Tracing**: Auto-instrument OpenAI, Anthropic, Google, Cohere, and more
- **Decorator-based Instrumentation**: Simple `@instrument()` decorator for custom functions
- **OpenTelemetry Native**: Built on OTLP for compatibility with any OTel backend
- **Typed Service Client**: Async helpers for prompts, retrieval, toolbox, guardrails, evals, and workers
- **Correlation Context**: Propagate interaction, trace, and agent lineage hints for diagnosis
- **Minimal Overhead**: Async batching, <0.5ms per span
- **Type Safe**: Full type hints and `py.typed` marker

## Installation

```bash
# Basic install
pip install taproot-sdk

# With auto-instrumentation for specific providers
pip install taproot-sdk[openai]
pip install taproot-sdk[anthropic]
pip install taproot-sdk[google]

# All LLM providers
pip install taproot-sdk[all]
```

## Quick Start

```python
import taproot_sdk as ev

# Initialize once at app startup
ev.init(
    project_id="my-project",
    api_url="https://your-taproot-backend.com",
    api_key="sk-...",
    auto_instrument=["openai", "anthropic"],  # Auto-trace LLM calls
)

# Decorate custom functions
@ev.instrument(spankind="retrieval")
def search_knowledge_base(query: str) -> list:
    return vector_db.search(query, top_k=5)

@ev.instrument(spankind="tool")
def create_ticket(title: str, description: str) -> dict:
    return api.create_ticket(title=title, description=description)

# LLM calls are automatically traced!
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)  # This is automatically traced
```

## Span Types

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

## Configuration Options

```python
ev.init(
    project_id="my-project",           # Required: Project identifier
    api_url="https://...",             # Required: Taproot backend URL
    api_key="sk-...",                  # Optional: API key for authentication
    auto_instrument=["openai"],        # Optional: LLM libraries to auto-instrument
    redact_by_default=True,            # Optional: Hash PII in traces (default: True)
    sampling_rate=1.0,                 # Optional: Trace sampling rate (default: 1.0)
    batch_size=512,                    # Optional: Spans per batch (default: 512)
    flush_interval_ms=5000,            # Optional: Max time between flushes (default: 5000)
)
```

## Decorator Options

```python
@ev.instrument(
    spankind="tool",                   # Span type (see table above)
    name="custom-span-name",           # Override function name
    ignore_inputs=False,               # Don't capture inputs
    ignore_outputs=False,              # Don't capture outputs
)
def my_function():
    pass
```

## Supported LLM Libraries

Auto-instrumentation is available for:

- **OpenAI** (`openai`) - ChatCompletion, Completion, Embeddings
- **Anthropic** (`anthropic`) - Messages, Completions
- **Google GenAI** (`google-generativeai`) - GenerativeModel
- **Vertex AI** (`vertexai`) - GenerativeModel
- **AWS Bedrock** (`boto3` bedrock-runtime)
- **Cohere** (`cohere`) - Chat, Generate, Embed
- **Mistral** (`mistralai`) - Chat

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/taproot-ai/taproot-sdk.git
cd taproot-sdk
uv pip install -e ".[dev,all]"

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## License

Apache 2.0
