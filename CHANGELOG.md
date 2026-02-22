# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- PII redaction support
- Configurable sampling rate
- Batch span export with gzip compression

[Unreleased]: https://github.com/taproot-ai/taproot-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/taproot-ai/taproot-sdk/releases/tag/v0.1.0
