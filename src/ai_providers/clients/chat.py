# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Chat client for OpenAI-compatible chat completions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ai_providers.clients.http import OpenAIHTTPClient
from ai_providers.config.providers import OPENAI_CONFIG
from ai_providers.models import (
    ChatMessage,
    ChatOptions,
    ChatResult,
    ChatStreamEvent,
    TokenUsage,
)


class ChatClient:
    def __init__(self, http: OpenAIHTTPClient) -> None:
        self._http = http

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        payload = _chat_payload(messages, options)
        data = self._http.post_json("/chat/completions", payload)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = _extract_content(message.get("content", ""))
        return ChatResult(raw=data, content=content, usage=_usage(data.get("usage")))

    def chat_stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> Iterator[ChatStreamEvent]:
        payload = {**_chat_payload(messages, options), "stream": True}
        content = ""
        for event in self._http.stream_json("/chat/completions", payload):
            choice = event.get("choices", [{}])[0]
            delta = _extract_content(choice.get("delta", {}).get("content", ""))
            if delta:
                content += delta
            yield ChatStreamEvent(content=content, delta=delta, raw=event)


def _chat_payload(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    options: ChatOptions | None,
) -> dict[str, Any]:
    options = options or ChatOptions()
    payload: dict[str, Any] = {
        "model": options.model or str(OPENAI_CONFIG["chat"]["model"]),
        "messages": [_message_to_dict(message) for message in messages],
        "temperature": (
            options.temperature
            if options.temperature is not None
            else OPENAI_CONFIG["chat"]["default_temperature"]
        ),
    }
    if options.max_tokens is not None:
        payload["max_tokens"] = options.max_tokens
    if options.response_format is not None:
        payload["response_format"] = options.response_format
    if options.tools:
        payload["tools"] = list(options.tools)
    return payload


def _message_to_dict(message: ChatMessage | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    return dict(message)


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    return str(content or "")


def _usage(raw: Any) -> TokenUsage | None:
    if not isinstance(raw, Mapping):
        return None
    return TokenUsage(
        input_tokens=int(raw.get("prompt_tokens", raw.get("input_tokens", 0))),
        output_tokens=int(raw.get("completion_tokens", raw.get("output_tokens", 0))),
        total_tokens=int(raw.get("total_tokens", 0)),
    )
