# -*- coding: utf-8 -*-
"""Focused contract tests for BaseChannel high-risk shared behavior."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from copaw.app.channels.base import (
    AudioContent,
    BaseChannel,
    ContentType,
    FileContent,
    RefusalContent,
    TextContent,
)


def _patch_channel_base_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid coupling tests to the real global config loader."""
    fake_tools = SimpleNamespace(builtin_tools={})
    fake_config = SimpleNamespace(tools=fake_tools)
    monkeypatch.setattr(
        "copaw.app.channels.base.load_config",
        lambda: fake_config,
    )


class _ContractChannel(BaseChannel):
    """Small test double for BaseChannel contract tests."""

    channel = "contract"

    def __init__(self, process, *, bot_prefix: str = "") -> None:
        super().__init__(process)
        self.bot_prefix = bot_prefix
        self.sent_messages: list[tuple[str, str, dict | None]] = []
        self.sent_media: list[tuple[str, object, dict | None]] = []

    @classmethod
    def from_env(cls, process, on_reply_sent=None):
        del on_reply_sent
        return cls(process)

    @classmethod
    def from_config(
        cls,
        process,
        config,
        on_reply_sent=None,
        **kwargs,
    ):
        del on_reply_sent, kwargs
        return cls(
            process,
            bot_prefix=getattr(config, "bot_prefix", ""),
        )

    def resolve_session_id(self, sender_id: str, channel_meta=None) -> str:
        if channel_meta and channel_meta.get("thread_id"):
            return f"{self.channel}:{channel_meta['thread_id']}"
        return super().resolve_session_id(sender_id, channel_meta)

    def build_agent_request_from_native(self, native_payload):
        raise NotImplementedError

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, to_handle: str, text: str, meta=None) -> None:
        self.sent_messages.append((to_handle, text, meta))

    async def send_media(self, to_handle: str, part, meta=None) -> None:
        self.sent_media.append((to_handle, part, meta))


def _make_request(
    *,
    text: str,
    session_id: str = "contract:user-1",
    user_id: str = "user-1",
    channel: str = "contract",
):
    return SimpleNamespace(
        session_id=session_id,
        user_id=user_id,
        channel=channel,
        input=[
            SimpleNamespace(
                content=[
                    TextContent(type=ContentType.TEXT, text=text),
                ],
            ),
        ],
    )


def test_get_debounce_key_uses_explicit_session_or_resolved_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)

    assert (
        channel.get_debounce_key({"session_id": "existing", "sender_id": "u1"})
        == "existing"
    )
    assert (
        channel.get_debounce_key(
            {
                "sender_id": "u1",
                "meta": {"thread_id": "thread-9"},
            },
        )
        == "contract:thread-9"
    )


def test_apply_no_text_debounce_buffers_then_flushes_with_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)
    session_id = "contract:user-1"
    file_part = FileContent(
        type=ContentType.FILE,
        filename="report.txt",
        file_url="file:///tmp/report.txt",
    )
    text_part = TextContent(type=ContentType.TEXT, text="summarize this")

    should_process, merged = channel._apply_no_text_debounce(
        session_id,
        [file_part],
    )
    assert should_process is False
    assert merged == []

    should_process, merged = channel._apply_no_text_debounce(
        session_id,
        [text_part],
    )
    assert should_process is True
    assert merged == [file_part, text_part]


def test_apply_no_text_debounce_audio_bypasses_waiting_for_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)
    session_id = "contract:user-1"
    file_part = FileContent(
        type=ContentType.FILE,
        filename="voice-note.txt",
        file_url="file:///tmp/voice-note.txt",
    )
    audio_part = AudioContent(
        type=ContentType.AUDIO,
        data="file:///tmp/voice-note.wav",
    )

    channel._apply_no_text_debounce(session_id, [file_part])
    should_process, merged = channel._apply_no_text_debounce(
        session_id,
        [audio_part],
    )

    assert should_process is True
    assert merged == [file_part, audio_part]


def test_extract_query_from_payload_supports_dict_and_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)
    dict_payload = {
        "content_parts": [
            TextContent(type=ContentType.TEXT, text="/stop now"),
        ],
    }
    request_payload = _make_request(text="hello from request")

    assert channel._extract_query_from_payload(dict_payload) == "/stop now"
    assert (
        channel._extract_query_from_payload(request_payload)
        == "hello from request"
    )


@pytest.mark.asyncio
async def test_send_content_parts_merges_text_refusal_and_media_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)
    file_part = FileContent(
        type=ContentType.FILE,
        filename="report.txt",
        file_url="file:///tmp/report.txt",
    )

    await channel.send_content_parts(
        "room-1",
        [
            TextContent(type=ContentType.TEXT, text="hello"),
            RefusalContent(type=ContentType.REFUSAL, refusal="cannot do that"),
            file_part,
        ],
        {"bot_prefix": "[bot]"},
    )

    assert channel.sent_messages == [
        (
            "room-1",
            "[bot]  hello\ncannot do that\n[File: file:///tmp/report.txt]",
            {"bot_prefix": "[bot]"},
        ),
    ]
    assert channel.sent_media == [("room-1", file_part, {"bot_prefix": "[bot]"})]


@pytest.mark.asyncio
async def test_consume_one_request_routes_non_control_messages_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process)
    request = _make_request(text="normal message")
    payload = {"content_parts": request.input[0].content}
    channel._workspace = object()
    channel._command_registry = SimpleNamespace(
        is_control_command=lambda query: query.startswith("/"),
    )
    channel._payload_to_request = lambda actual_payload: request
    channel._consume_with_tracker = AsyncMock()
    channel._run_process_loop = AsyncMock()

    await channel._consume_one_request(payload)

    channel._consume_with_tracker.assert_awaited_once_with(request, payload)
    channel._run_process_loop.assert_not_called()


@pytest.mark.asyncio
async def test_consume_one_request_processes_control_messages_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_channel_base_config(monkeypatch)

    async def _noop_process(_request):
        if False:
            yield None

    channel = _ContractChannel(_noop_process, bot_prefix="[bot]")
    request = _make_request(text="/stop")
    payload = {
        "content_parts": request.input[0].content,
        "meta": {"trace_id": "t-1"},
        "session_webhook": "https://example.com/webhook",
    }
    channel._workspace = object()
    channel._command_registry = SimpleNamespace(
        is_control_command=lambda query: query.startswith("/"),
    )
    channel._payload_to_request = lambda actual_payload: request
    channel._consume_with_tracker = AsyncMock()
    channel._before_consume_process = AsyncMock()
    channel._run_process_loop = AsyncMock()

    await channel._consume_one_request(payload)

    channel._consume_with_tracker.assert_not_called()
    channel._before_consume_process.assert_awaited_once_with(request)
    channel._run_process_loop.assert_awaited_once()
    _, to_handle, send_meta = channel._run_process_loop.await_args.args
    assert to_handle == "user-1"
    assert send_meta == {
        "trace_id": "t-1",
        "session_webhook": "https://example.com/webhook",
        "bot_prefix": "[bot]",
    }
    assert request.channel_meta == {
        "trace_id": "t-1",
        "session_webhook": "https://example.com/webhook",
    }
