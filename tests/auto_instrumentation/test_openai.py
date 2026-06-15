"""Tests for OpenAI auto-instrumentation wiring."""

from unittest.mock import Mock, patch

from taproot_sdk.auto_instrument import (
    INSTRUMENTORS,
    get_instrumented_libraries,
    setup_auto_instrumentation,
    uninstrument,
    uninstrument_all,
)


def test_setup_openai_loads_configured_instrumentor():
    uninstrument_all()
    instrumentor = Mock()

    with patch("taproot_sdk.auto_instrument._load_instrumentor", return_value=instrumentor) as load:
        assert setup_auto_instrumentation(["openai"]) == ["openai"]

    load.assert_called_once_with(INSTRUMENTORS["openai"])
    instrumentor.instrument.assert_called_once_with()
    assert get_instrumented_libraries() == ["openai"]


def test_setup_openai_is_idempotent():
    uninstrument_all()
    instrumentor = Mock()

    with patch("taproot_sdk.auto_instrument._load_instrumentor", return_value=instrumentor):
        assert setup_auto_instrumentation(["openai"]) == ["openai"]
        assert setup_auto_instrumentation(["openai"]) == ["openai"]

    instrumentor.instrument.assert_called_once_with()


def test_uninstrument_openai_calls_configured_instrumentor():
    uninstrument_all()
    instrumentor = Mock()

    with patch("taproot_sdk.auto_instrument._load_instrumentor", return_value=instrumentor):
        setup_auto_instrumentation(["openai"])
        assert uninstrument(["openai"]) == ["openai"]

    instrumentor.uninstrument.assert_called_once_with()
    assert get_instrumented_libraries() == []
