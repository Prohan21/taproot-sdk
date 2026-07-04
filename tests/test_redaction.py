"""WO-013 T1: real PII/secret redaction for span inputs/outputs."""

from __future__ import annotations

import hashlib

from taproot_sdk._redaction import scrub_structured, scrub_text


def _token(value: str) -> str:
    return f"redacted:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


class TestScrubText:
    def test_openai_style_key(self) -> None:
        out = scrub_text("using sk-live-abc123XYZ789 for auth")
        assert "sk-live-abc123XYZ789" not in out
        assert _token("sk-live-abc123XYZ789") in out

    def test_aws_access_key(self) -> None:
        out = scrub_text("key AKIAIOSFODNN7EXAMPLE end")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "redacted:" in out

    def test_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQs"
        out = scrub_text(f"Bearer {jwt}")
        assert jwt not in out
        assert "redacted:" in out

    def test_email(self) -> None:
        out = scrub_text("contact a@b.com please")
        assert "a@b.com" not in out
        assert "redacted:" in out

    def test_ssn(self) -> None:
        out = scrub_text("ssn is 123-45-6789 ok")
        assert "123-45-6789" not in out

    def test_credit_card(self) -> None:
        out = scrub_text("card 4111 1111 1111 1111 exp 12/28")
        assert "4111 1111 1111 1111" not in out

    def test_sensitive_key_value_pair_in_text(self) -> None:
        out = scrub_text("connect with password=hunter2 now")
        assert "hunter2" not in out
        assert "password=" in out

    def test_token_is_stable_and_non_reversible(self) -> None:
        a = scrub_text("sk-live-abc123XYZ789")
        b = scrub_text("sk-live-abc123XYZ789")
        assert a == b
        assert a == _token("sk-live-abc123XYZ789")

    def test_clean_text_unchanged(self) -> None:
        text = "the quick brown fox ran 42 times"
        assert scrub_text(text) == text


class TestScrubStructured:
    def test_sensitive_keys_redacted(self) -> None:
        out = scrub_structured({"api_key": "plainvalue", "query": "hello"})
        assert out["api_key"] == _token("plainvalue")
        assert out["query"] == "hello"

    def test_sensitive_key_case_insensitive(self) -> None:
        out = scrub_structured({"Authorization": "Basic abc", "PASSWORD": "x"})
        assert out["Authorization"].startswith("redacted:")
        assert out["PASSWORD"].startswith("redacted:")

    def test_nested_structures(self) -> None:
        out = scrub_structured(
            {
                "outer": {"token": "tok-123", "safe": 1},
                "items": [{"secret": "s3cr3t"}, "a@b.com"],
            }
        )
        assert out["outer"]["token"].startswith("redacted:")
        assert out["outer"]["safe"] == 1
        assert out["items"][0]["secret"].startswith("redacted:")
        assert "a@b.com" not in out["items"][1]

    def test_non_string_value_under_sensitive_key(self) -> None:
        out = scrub_structured({"credentials": {"user": "u", "password": "p"}})
        assert isinstance(out["credentials"], str)
        assert out["credentials"].startswith("redacted:")

    def test_tuples_preserved_as_tuples(self) -> None:
        out = scrub_structured({"values": (1, "plain", "x@y.io")})
        assert isinstance(out["values"], tuple)
        assert out["values"][0] == 1
        assert "x@y.io" not in out["values"][2]

    def test_does_not_mutate_input(self) -> None:
        original = {"api_key": "abc", "nested": {"password": "p"}}
        scrub_structured(original)
        assert original == {"api_key": "abc", "nested": {"password": "p"}}

    def test_scalars_pass_through(self) -> None:
        assert scrub_structured(42) == 42
        assert scrub_structured(None) is None
        assert scrub_structured(True) is True
