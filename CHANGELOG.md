# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Real PII/secret redaction for `@instrument` span inputs/outputs, gated by
  `redact_by_default` (previously accepted but a no-op). Sensitive keys and
  secret/PII-shaped values are replaced with stable `redacted:<hash>` tokens;
  correlation/interaction ids and `ev.meta.*` are never scrubbed. Per-function
  override via `@instrument(redact=...)`. **Behavior change:** redaction is
  on by default — pass `redact_by_default=False` for the previous plaintext
  behavior. LLM auto-instrumentor spans are NOT covered (see README).

### Fixed

- Removed the `[0.1.0]` "PII redaction support" claim: the flag existed but
  performed no redaction in that release.

## [0.1.0] - 2024-XX-XX

### Added

- Initial release of Taproot SDK
- `ev.init()` for SDK initialization with OpenTelemetry backend
- `@ev.instrument()` decorator for tracing custom functions
- Auto-instrumentation support for LLM libraries:
  - OpenAI
  - Anthropic
  - Google Generative AI
  - Cohere
  - Vertex AI
  - AWS Bedrock
  - Mistral AI
- 9 span types: workflow, agent, chain, tool, retrieval, embedding, completion, chat, rerank
- Async function support
- Input/output capture with size limits
- Configurable sampling rate
- Batch span export with gzip compression

[Unreleased]: https://github.com/taproot-ai/taproot-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/taproot-ai/taproot-sdk/releases/tag/v0.1.0
