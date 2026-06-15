"""
Taproot SDK - Instrumentation for LLM observability.

Usage:
    import taproot_sdk as ev

    ev.init(
        project_id="my-project",
        api_url="https://your-backend.com",
        api_key="sk-...",
        auto_instrument=["openai", "anthropic"],
    )

    @ev.instrument(spankind="tool")
    def my_function():
        pass

Prompt management and serving:
    client = ev.TaprootClient(
        base_url="https://gateway.taproot.dev",
        api_key="your-api-key",
        project_id="my-project",
    )
    prompt = await client.get_prompt("welcome-email")
    rendered = prompt.render(user_name="Alice")
"""

from taproot_sdk._context import (
    TaprootActorRef,
    TaprootInteractionContext,
    clear_interaction_context,
    get_interaction_context,
    merge_propagation_headers,
    propagation_headers,
    reset_interaction_context,
    set_interaction_context,
)
from taproot_sdk.client import TaprootClient
from taproot_sdk.core import get_tracer, init, is_initialized, shutdown
from taproot_sdk.decorators import instrument
from taproot_sdk.evals import (
    EvalAssertionError,
    EvalResult,
    Experiment,
    GoldenDataset,
    GoldenDatasetItem,
    GoldenDatasetVersion,
    PaginatedList,
    RunHandle,
    TestConfiguration,
    assert_eval,
)
from taproot_sdk.exceptions import (
    AuthenticationError,
    ConflictError,
    PromptNotFoundError,
    RateLimitError,
    ServerError,
    TaprootAPIError,
    TaprootError,
    ValidationError,
)
from taproot_sdk.guardrails import (
    AnalyticsSummary,
    CheckResult,
    GuardrailConfig,
    GuardrailResponse,
    ScannerOverride,
    ScannerSignal,
)
from taproot_sdk.instrument import instrument_app
from taproot_sdk.prompts import MissingVariableError, PromptResponse
from taproot_sdk.retrieval import (
    AccessGrant,
    AccessGranted,
    AccessList,
    AccessRevoked,
    BatchCancelled,
    BatchCreated,
    BatchJobList,
    BatchStatus,
    ChunkInfo,
    ChunkList,
    ChunksDeleted,
    ChunksUploaded,
    DocumentDeleted,
    DocumentDetail,
    DocumentInfo,
    DocumentList,
    IndexParams,
    IngestionJob,
    JobCancelled,
    JobDetail,
    JobList,
    JobSummary,
    QueryHit,
    QueryResponse,
    StoreCreated,
    StoreDeleted,
    StoreInfo,
    StoreList,
    StoreStatistics,
)
from taproot_sdk.toolbox import (
    CredentialInfo,
    CredentialList,
    ImportResult,
    InvocationResult,
    MCPServerInfo,
    MCPServerList,
    ToolInfo,
    ToolList,
)

__version__ = "0.1.0"

__all__ = [
    # Client
    "TaprootClient",
    # Core functions
    "init",
    "shutdown",
    "get_tracer",
    "is_initialized",
    # Decorators
    "instrument",
    # App instrumentation
    "instrument_app",
    # TAP-38 interaction propagation
    "TaprootActorRef",
    "TaprootInteractionContext",
    "clear_interaction_context",
    "get_interaction_context",
    "merge_propagation_headers",
    "propagation_headers",
    "reset_interaction_context",
    "set_interaction_context",
    # Exceptions
    "TaprootError",
    "TaprootAPIError",
    "PromptNotFoundError",
    "AuthenticationError",
    "ConflictError",
    "RateLimitError",
    "ServerError",
    "ValidationError",
    # Prompts
    "PromptResponse",
    "MissingVariableError",
    # Evals
    "EvalResult",
    "RunHandle",
    "assert_eval",
    "EvalAssertionError",
    "GoldenDataset",
    "GoldenDatasetItem",
    "GoldenDatasetVersion",
    "TestConfiguration",
    "Experiment",
    "PaginatedList",
    # Guardrails
    "AnalyticsSummary",
    "CheckResult",
    "GuardrailConfig",
    "GuardrailResponse",
    "ScannerOverride",
    "ScannerSignal",
    # Retrieval
    "IndexParams",
    "StoreInfo",
    "StoreList",
    "StoreCreated",
    "StoreDeleted",
    "StoreStatistics",
    "AccessGrant",
    "AccessGranted",
    "AccessList",
    "AccessRevoked",
    "QueryHit",
    "QueryResponse",
    "IngestionJob",
    "JobDetail",
    "JobList",
    "JobSummary",
    "JobCancelled",
    "BatchCreated",
    "BatchStatus",
    "BatchJobList",
    "BatchCancelled",
    "DocumentInfo",
    "DocumentDetail",
    "DocumentList",
    "DocumentDeleted",
    "ChunkInfo",
    "ChunkList",
    "ChunksUploaded",
    "ChunksDeleted",
    # ToolBox
    "CredentialInfo",
    "CredentialList",
    "ImportResult",
    "InvocationResult",
    "MCPServerInfo",
    "MCPServerList",
    "ToolInfo",
    "ToolList",
    # Version
    "__version__",
]
