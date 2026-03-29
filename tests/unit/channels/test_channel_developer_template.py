# -*- coding: utf-8 -*-
"""Minimal sample tests for community channel authors.

This file is intentionally small and opinionated. It acts as a living
template for developers adding a custom channel: define a tiny test double,
verify `from_config`, verify session routing, and verify outbound send logic.
"""
from __future__ import annotations

from types import SimpleNamespace

from copaw.app.channels.base import BaseChannel


def _patch_channel_base_config(monkeypatch) -> None:
    fake_tools = SimpleNamespace(builtin_tools={})
    fake_config = SimpleNamespace(tools=fake_tools)
    monkeypatch.setattr(
        "copaw.app.channels.base.load_config",
        lambda: fake_config,
    )


class TemplateCommunityChannel(BaseChannel):
    """Small reference implementation used only by tests."""

    channel = "template_community"

    def __init__(
        self,
        process,
        *,
        enabled: bool = True,
        bot_prefix: str = "",
        on_reply_sent=None,
    ) -> None:
        super().__init__(process, on_reply_sent=on_reply_sent)
        self.enabled = enabled
        self.bot_prefix = bot_prefix
        self.started = False
        self.stopped = False
        self.sent_messages: list[tuple[str, str, dict | None]] = []

    @classmethod
    def from_env(cls, process, on_reply_sent=None):
        return cls(process=process, on_reply_sent=on_reply_sent)

    @classmethod
    def from_config(cls, process, config, on_reply_sent=None, **kwargs):
        del kwargs
        return cls(
            process=process,
            enabled=config.enabled,
            bot_prefix=config.bot_prefix,
            on_reply_sent=on_reply_sent,
        )

    def resolve_session_id(self, sender_id: str, channel_meta=None) -> str:
        if channel_meta and channel_meta.get("thread_id"):
            return f"{self.channel}:{channel_meta['thread_id']}"
        return super().resolve_session_id(sender_id, channel_meta)

    def build_agent_request_from_native(self, native_payload):
        raise NotImplementedError

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, to_handle: str, text: str, meta=None) -> None:
        self.sent_messages.append((to_handle, text, meta))


def test_template_from_config_maps_channel_fields(monkeypatch) -> None:
    _patch_channel_base_config(monkeypatch)

    channel = TemplateCommunityChannel.from_config(
        process=lambda _: None,
        config=SimpleNamespace(enabled=True, bot_prefix="[community]"),
        on_reply_sent="reply-hook",
    )

    assert channel.enabled is True
    assert channel.bot_prefix == "[community]"
    assert channel._on_reply_sent == "reply-hook"


def test_template_resolve_session_id_prefers_thread_context(
    monkeypatch,
) -> None:
    _patch_channel_base_config(monkeypatch)
    channel = TemplateCommunityChannel(lambda _: None)

    assert (
        channel.resolve_session_id("user-1", {"thread_id": "thread-9"})
        == "template_community:thread-9"
    )
    assert (
        channel.resolve_session_id("user-1", None)
        == "template_community:user-1"
    )


async def test_template_send_captures_outbound_message(monkeypatch) -> None:
    _patch_channel_base_config(monkeypatch)
    channel = TemplateCommunityChannel(lambda _: None)

    await channel.send("room-1", "hello template", {"trace_id": "abc"})

    assert channel.sent_messages == [
        ("room-1", "hello template", {"trace_id": "abc"}),
    ]
