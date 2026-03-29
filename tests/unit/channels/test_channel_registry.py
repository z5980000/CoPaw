# -*- coding: utf-8 -*-
"""Regression tests for channel registry behavior."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from copaw.app.channels import registry
from copaw.app.channels.base import BaseChannel


class _BuiltinStubChannel(BaseChannel):
    """Simple BaseChannel subtype used as a registry sentinel."""

    channel = "console"

    @classmethod
    def from_env(cls, process, on_reply_sent=None):
        raise NotImplementedError

    @classmethod
    def from_config(cls, process, config, on_reply_sent=None, **kwargs):
        raise NotImplementedError

    def build_agent_request_from_native(self, native_payload):
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def send(self, to_handle: str, text: str, meta=None) -> None:
        raise NotImplementedError


def test_builtin_channel_cache_can_be_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_loader():
        calls["count"] += 1
        return {"console": _BuiltinStubChannel}

    registry.clear_builtin_channel_cache()
    monkeypatch.setattr(registry, "_load_builtin_channels", fake_loader)

    first = registry._get_cached_builtin_channels()
    second = registry._get_cached_builtin_channels()
    registry.clear_builtin_channel_cache()
    third = registry._get_cached_builtin_channels()

    assert first == {"console": _BuiltinStubChannel}
    assert second == {"console": _BuiltinStubChannel}
    assert third == {"console": _BuiltinStubChannel}
    assert calls["count"] == 2


def test_get_channel_registry_merges_custom_channels(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry.clear_builtin_channel_cache()
    monkeypatch.setattr(
        registry,
        "_get_cached_builtin_channels",
        lambda: {"console": _BuiltinStubChannel},
    )
    monkeypatch.setattr(registry, "CUSTOM_CHANNELS_DIR", tmp_path)

    module = tmp_path / "sample_custom.py"
    module.write_text(
        "\n".join(
            [
                "from copaw.app.channels.base import BaseChannel",
                "",
                "class SampleCustomChannel(BaseChannel):",
                "    channel = 'sample_custom'",
                "",
                "    @classmethod",
                "    def from_env(cls, process, on_reply_sent=None):",
                "        raise NotImplementedError",
                "",
                "    @classmethod",
                "    def from_config(",
                "        cls, process, config, on_reply_sent=None, **kwargs",
                "    ):",
                "        raise NotImplementedError",
                "",
                "    def build_agent_request_from_native(self, native_payload):",
                "        raise NotImplementedError",
                "",
                "    async def start(self):",
                "        raise NotImplementedError",
                "",
                "    async def stop(self):",
                "        raise NotImplementedError",
                "",
                "    async def send(self, to_handle, text, meta=None):",
                "        raise NotImplementedError",
            ],
        ),
        encoding="utf-8",
    )

    channel_registry = registry.get_channel_registry()

    assert channel_registry["console"] is _BuiltinStubChannel
    assert "sample_custom" in channel_registry
    assert issubclass(channel_registry["sample_custom"], BaseChannel)


def test_register_custom_channel_routes_warns_for_non_api_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    monkeypatch.setattr(registry, "CUSTOM_CHANNELS_DIR", tmp_path)
    monkeypatch.setattr(
        registry.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    module = tmp_path / "bad_routes.py"
    module.write_text(
        "\n".join(
            [
                "from types import SimpleNamespace",
                "",
                "def register_app_routes(app):",
                "    app.routes.append(SimpleNamespace(path='/api/ok'))",
                "    app.routes.append(SimpleNamespace(path='/not-api'))",
            ],
        ),
        encoding="utf-8",
    )

    app = SimpleNamespace(routes=[SimpleNamespace(path="/existing")])

    registry.register_custom_channel_routes(app)

    assert any("/not-api" in message for message in warnings)
    assert any(route.path == "/api/ok" for route in app.routes)


def test_load_builtin_channels_raises_when_required_console_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry,
        "_BUILTIN_SPECS",
        {"console": (".missing_console", "MissingConsoleChannel")},
    )

    def fake_import_module(name, package=None):
        del package
        raise ImportError(name)

    monkeypatch.setattr(registry.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError):
        registry._load_builtin_channels()
