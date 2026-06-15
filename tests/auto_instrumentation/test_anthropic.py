"""Tests for Anthropic auto-instrumentation wiring."""

from unittest.mock import Mock, patch

from taproot_sdk.auto_instrument import INSTRUMENTORS, setup_auto_instrumentation, uninstrument_all


def test_setup_anthropic_loads_configured_instrumentor():
    uninstrument_all()
    instrumentor = Mock()

    with patch("taproot_sdk.auto_instrument._load_instrumentor", return_value=instrumentor) as load:
        assert setup_auto_instrumentation(["anthropic"]) == ["anthropic"]

    load.assert_called_once_with(INSTRUMENTORS["anthropic"])
    instrumentor.instrument.assert_called_once_with()


def test_setup_anthropic_ignores_missing_optional_package():
    uninstrument_all()

    with patch("taproot_sdk.auto_instrument._load_instrumentor", side_effect=ImportError):
        assert setup_auto_instrumentation(["anthropic"]) == []
