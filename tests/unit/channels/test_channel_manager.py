# -*- coding: utf-8 -*-
"""Focused unit tests for the channel manager public layer."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from copaw.app.channels.base import BaseChannel, ContentType, TextContent
from copaw.app.channels.manager import ChannelManager, _process_batch


def _patch_channel_base_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid coupling tests to the real global config loader."""
    fake_tools = SimpleNamespace(builtin_tools={})
    fake_config = SimpleNamespace(tools=fake_tools)
    monkeypatch.setattr(
        "copaw.app.channels.base.load_config",
        lambda: fake_config,
    )


class _RecorderChannel(BaseChannel):
    """Small test double for ChannelManager tests."""

    channel = "console"
    last_from_config: dict | None = None

    def __init__(
        self,
        process,
        *,
        enabled: bool = True,
        on_reply_sent=None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        workspace_dir=None,
    ) -> None:
        super().__init__(
            process,
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
        )
        self.enabled = enabled
        self.workspace_dir = workspace_dir
        self.started = False
        self.stopped = False
        self.sent_messages: list[tuple[str, str, dict | None]] = []

    @classmethod
    def from_env(cls, process, on_reply_sent=None):
        return cls(process, on_reply_sent=on_reply_sent)

    @classmethod
    def from_config(
        cls,
        process,
        config,
        on_reply_sent=None,
        show_tool_details: bool = True,
        filter_tool_messages: bool = False,
        filter_thinking: bool = False,
        workspace_dir=None,
    ):
        cls.last_from_config = {
            "config": config,
            "on_reply_sent": on_reply_sent,
            "show_tool_details": show_tool_details,
            "filter_tool_messages": filter_tool_messages,
            "filter_thinking": filter_thinking,
            "workspace_dir": workspace_dir,
        }
        return cls(
            process,
            enabled=getattr(config, "enabled", True),
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
            workspace_dir=workspace_dir,
        )

    def build_agent_request_from_native(self, native_payload):
        raise NotImplementedError

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, to_handle: str, text: str, meta=None) -> None:
        self.sent_messages.append((to_handle, text, meta))


class _MinimalSignatureChannel(_RecorderChannel):
    """Simulates a community channel with a narrow from_config signature."""

    channel = "sample_custom"
    last_from_config: dict | None = None

    @classmethod
    def from_config(
        cls,
        process,
        config,
        on_reply_sent=None,
    ):
        cls.last_from_config = {
            "config": config,
            "on_reply_sent": on_reply_sent,
        }
        return cls(
            process,
            enabled=getattr(config, "enabled", True),
            on_reply_sent=on_reply_sent,
        )


class _ExplodingChannel(_RecorderChannel):
    """Simulates a broken channel implementation."""

    channel = "broken"

    @classmethod
    def from_config(cls, process, config, on_reply_sent=None, **kwargs):
        del process, config, on_reply_sent, kwargs
        raise RuntimeError("boom")


class _StartFailChannel(_RecorderChannel):
    """Used to verify replace_channel failure semantics."""

    async def start(self) -> None:
        raise RuntimeError("start failed")


@pytest.mark.asyncio
async def test_from_config_loads_enabled_builtin_and_custom_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)
    monkeypatch.setattr(
        "copaw.app.channels.manager.get_available_channels",
        lambda: ("console", "sample_custom"),
    )
    monkeypatch.setattr(
        "copaw.app.channels.manager.get_channel_registry",
        lambda: {
            "console": _RecorderChannel,
            "sample_custom": _MinimalSignatureChannel,
        },
    )

    channels_cfg = SimpleNamespace(
        console=SimpleNamespace(
            enabled=True,
            filter_tool_messages=True,
            filter_thinking=True,
        ),
        __pydantic_extra__={
            "sample_custom": {
                "enabled": True,
                "bot_prefix": "[community]",
            },
        },
    )
    config = SimpleNamespace(
        channels=channels_cfg,
        show_tool_details=False,
    )

    manager = ChannelManager.from_config(
        process=AsyncMock(),
        config=config,
        on_last_dispatch="dispatch-hook",
        workspace_dir="/tmp/agent-a",
    )

    assert [channel.channel for channel in manager.channels] == [
        "console",
        "sample_custom",
    ]
    assert _RecorderChannel.last_from_config == {
        "config": channels_cfg.console,
        "on_reply_sent": "dispatch-hook",
        "show_tool_details": False,
        "filter_tool_messages": True,
        "filter_thinking": True,
        "workspace_dir": "/tmp/agent-a",
    }
    assert _MinimalSignatureChannel.last_from_config is not None
    assert (
        _MinimalSignatureChannel.last_from_config["on_reply_sent"]
        == "dispatch-hook"
    )
    normalized_custom_cfg = _MinimalSignatureChannel.last_from_config[
        "config"
    ]
    assert normalized_custom_cfg.enabled is True
    assert normalized_custom_cfg.bot_prefix == "[community]"
    assert normalized_custom_cfg.filter_tool_messages is False


def test_from_config_skips_broken_channels_without_blocking_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)
    monkeypatch.setattr(
        "copaw.app.channels.manager.get_available_channels",
        lambda: ("console", "broken"),
    )
    monkeypatch.setattr(
        "copaw.app.channels.manager.get_channel_registry",
        lambda: {
            "console": _RecorderChannel,
            "broken": _ExplodingChannel,
        },
    )

    config = SimpleNamespace(
        channels=SimpleNamespace(
            console=SimpleNamespace(enabled=True),
            broken=SimpleNamespace(enabled=True),
        ),
        show_tool_details=True,
    )

    manager = ChannelManager.from_config(
        process=AsyncMock(),
        config=config,
    )

    assert [channel.channel for channel in manager.channels] == ["console"]


@pytest.mark.asyncio
async def test_replace_channel_starts_new_before_stopping_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)
    events: list[str] = []

    class _OrderedOldChannel(_RecorderChannel):
        channel = "console"

        async def stop(self) -> None:
            events.append("old-stop")
            await super().stop()

    class _OrderedNewChannel(_RecorderChannel):
        channel = "console"

        async def start(self) -> None:
            events.append("new-start")
            await super().start()

    async def _noop_process(_request):
        if False:
            yield None

    old_channel = _OrderedOldChannel(_noop_process)
    manager = ChannelManager([old_channel])

    new_channel = _OrderedNewChannel(_noop_process)
    await manager.replace_channel(new_channel)

    assert events == ["new-start", "old-stop"]
    assert manager.channels == [new_channel]
    assert old_channel.stopped is True
    assert new_channel.started is True
    assert new_channel._enqueue is not None


@pytest.mark.asyncio
async def test_replace_channel_keeps_old_channel_when_new_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    old_channel = _RecorderChannel(_noop_process)
    manager = ChannelManager([old_channel])

    with pytest.raises(RuntimeError, match="start failed"):
        await manager.replace_channel(_StartFailChannel(_noop_process))

    assert manager.channels == [old_channel]
    assert old_channel.stopped is False


@pytest.mark.asyncio
async def test_process_batch_merges_native_payloads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _RecorderChannel(_noop_process)
    channel._consume_one_request = AsyncMock()

    first = {
        "channel_id": "console",
        "sender_id": "user-1",
        "content_parts": [
            TextContent(type=ContentType.TEXT, text="hello"),
        ],
        "meta": {"conversation_id": "conv-1"},
    }
    second = {
        "channel_id": "console",
        "sender_id": "user-1",
        "content_parts": [
            TextContent(type=ContentType.TEXT, text="world"),
        ],
        "meta": {"message_id": "msg-2"},
    }

    await _process_batch(channel, [first, second])

    channel._consume_one_request.assert_awaited_once()
    merged = channel._consume_one_request.await_args.args[0]
    assert [part.text for part in merged["content_parts"]] == [
        "hello",
        "world",
    ]
    assert merged["meta"]["conversation_id"] == "conv-1"
    assert merged["meta"]["message_id"] == "msg-2"
