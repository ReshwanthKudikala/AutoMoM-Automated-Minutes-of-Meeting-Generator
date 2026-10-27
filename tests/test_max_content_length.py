"""Focused tests for defensive MAX_CONTENT_LENGTH parsing."""

from __future__ import annotations

import logging

from config import DEFAULT_MAX_CONTENT_LENGTH, parse_max_content_length


def test_parse_max_content_length_default_when_absent():
    assert parse_max_content_length(None) == DEFAULT_MAX_CONTENT_LENGTH
    assert parse_max_content_length("") == DEFAULT_MAX_CONTENT_LENGTH
    assert parse_max_content_length("   ") == DEFAULT_MAX_CONTENT_LENGTH


def test_parse_max_content_length_valid_integer():
    assert parse_max_content_length("1048576") == 1048576
    assert parse_max_content_length(" 2048 ") == 2048
    assert parse_max_content_length(4096) == 4096


def test_parse_max_content_length_invalid_falls_back_without_logging_raw_value(
    caplog,
):
    with caplog.at_level(logging.WARNING, logger="config"):
        assert parse_max_content_length("not-a-number") == DEFAULT_MAX_CONTENT_LENGTH
        assert parse_max_content_length("12.5") == DEFAULT_MAX_CONTENT_LENGTH
        assert parse_max_content_length("-1") == DEFAULT_MAX_CONTENT_LENGTH

    assert any(
        "Invalid MAX_CONTENT_LENGTH" in record.getMessage()
        or "Negative MAX_CONTENT_LENGTH" in record.getMessage()
        for record in caplog.records
    )
    # Must not echo the malformed environment contents into logs.
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "not-a-number" not in joined
    assert "12.5" not in joined
