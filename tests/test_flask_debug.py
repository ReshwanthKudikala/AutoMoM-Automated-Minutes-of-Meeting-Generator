"""Unit tests for FLASK_DEBUG configuration."""

from __future__ import annotations

import logging

import pytest

from config import parse_bool_env


@pytest.mark.parametrize(
    "raw",
    [None, "", "   "],
)
def test_parse_bool_env_unset_defaults_to_false(raw):
    assert parse_bool_env(raw, default=False, env_name="FLASK_DEBUG") is False


@pytest.mark.parametrize(
    "raw",
    ["1", "true", "TRUE", "Yes", "on", " On "],
)
def test_parse_bool_env_truthy_values(raw):
    assert parse_bool_env(raw, default=False, env_name="FLASK_DEBUG") is True


@pytest.mark.parametrize(
    "raw",
    ["0", "false", "FALSE", "No", "off", " Off "],
)
def test_parse_bool_env_falsy_values(raw):
    assert parse_bool_env(raw, default=True, env_name="FLASK_DEBUG") is False


def test_parse_bool_env_invalid_falls_back_without_logging_raw_value(caplog):
    with caplog.at_level(logging.WARNING, logger="config"):
        assert parse_bool_env("maybe", default=False, env_name="FLASK_DEBUG") is False
        assert parse_bool_env("2", default=False, env_name="FLASK_DEBUG") is False

    assert any("Invalid FLASK_DEBUG" in record.getMessage() for record in caplog.records)
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "maybe" not in joined
    assert "2" not in joined


def test_create_app_debug_disabled_by_default(monkeypatch):
    monkeypatch.setattr("main.FLASK_DEBUG", False)
    from main import create_app

    app = create_app()
    assert app.debug is False
    assert app.config["DEBUG"] is False
