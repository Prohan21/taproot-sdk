"""Tests for guardrails response models, focused on the WO-005 degraded-mode contract."""

import dataclasses

import pytest

from taproot_sdk.guardrails.models import GuardrailResponse, ScannerSignal


def _full_response_payload() -> dict:
    return {
        "verdict": "BLOCK",
        "signals": [
            {
                "scanner_id": "prompt_injection",
                "scanner_version": "1.0",
                "is_valid": False,
                "score": 0.99,
                "labels": ["injection"],
                "reasoning": None,
                "latency_ms": 12.5,
            }
        ],
        "policy_signals": [{"segment_id": "seg-1", "violated": True, "reasoning": "policy hit"}],
        "blocked_by": "prompt_injection",
        "block_reasons": ["prompt injection detected"],
        "company_policy_version": "3",
        "project_policy_version": "1",
        "total_latency_ms": 45.0,
        "request_id": "req-123",
        "degraded": True,
        "unevaluated": ["llm_judge", "seg-2"],
    }


class TestDegradedSignaling:
    def test_from_api_response_parses_degraded_and_unevaluated(self):
        response = GuardrailResponse.from_api_response(_full_response_payload())

        assert response.degraded is True
        assert response.unevaluated == ("llm_judge", "seg-2")

    def test_defaults_false_and_empty_when_server_omits_fields(self):
        payload = _full_response_payload()
        del payload["degraded"]
        del payload["unevaluated"]

        response = GuardrailResponse.from_api_response(payload)

        assert response.degraded is False
        assert response.unevaluated == ()

    def test_constructor_defaults(self):
        response = GuardrailResponse(verdict="ALLOW")

        assert response.degraded is False
        assert response.unevaluated == ()

    def test_frozen_dataclass_rejects_mutation(self):
        response = GuardrailResponse.from_api_response(_full_response_payload())

        with pytest.raises(dataclasses.FrozenInstanceError):
            response.degraded = False  # type: ignore[misc]


class TestFromApiResponse:
    def test_full_payload_round_trip(self):
        response = GuardrailResponse.from_api_response(_full_response_payload())

        assert response.verdict == "BLOCK"
        assert response.is_blocked is True
        assert response.is_allowed is False
        assert response.blocked_by == "prompt_injection"
        assert response.block_reasons == ("prompt injection detected",)
        assert response.request_id == "req-123"
        assert len(response.signals) == 1
        assert isinstance(response.signals[0], ScannerSignal)
        assert response.flagged_scanners == response.signals
        assert len(response.violated_policies) == 1

    def test_minimal_payload(self):
        response = GuardrailResponse.from_api_response({"verdict": "ALLOW"})

        assert response.is_allowed is True
        assert response.signals == ()
        assert response.policy_signals == ()
        assert response.redactions is None
        assert response.degraded is False
        assert response.unevaluated == ()
