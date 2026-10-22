# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Public data models for the provider facade."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, TypeAlias

ChatRole: TypeAlias = Literal["system", "user", "assistant"]
ResponseFormat: TypeAlias = Literal["text", "json", "verbose_json", "srt", "vtt"]
TimestampGranularity: TypeAlias = Literal["word", "segment"]

AudioFileInput: TypeAlias = str | Path | bytes | bytearray | BinaryIO
ChatMessageContent: TypeAlias = str | Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class OpenAIOptions:
    api_key: str
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class CreateAIOptions:
    openai: OpenAIOptions


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: ChatMessageContent


@dataclass(frozen=True)
class ChatOptions:
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Mapping[str, Any] | None = None
    tools: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    raw: Mapping[str, Any]
    content: str
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ChatStreamEvent:
    content: str
    delta: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TranscribeOptions:
    model: str | None = None
    language: str | None = None
    response_format: ResponseFormat | None = None
    prompt: str | None = None
    temperature: float | None = None
    # Only honoured with response_format="verbose_json" by models that support it
    # (e.g. whisper-1, or Whisper models on OpenAI-compatible hosts such as Groq).
    timestamp_granularities: Sequence[TimestampGranularity] = field(default_factory=tuple)


@dataclass(frozen=True)
class TranscriptionToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionSegment:
    text: str
    start: float
    end: float
    tokens: Sequence[TranscriptionToken] = field(default_factory=tuple)
    # Whisper decoding diagnostics, present in verbose_json responses.
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: Sequence[TranscriptionSegment] = field(default_factory=tuple)
    words: Sequence[TranscriptionWord] = field(default_factory=tuple)


@dataclass(frozen=True)
class EmbedOptions:
    model: str | None = None
    dimensions: int | None = None


EmbedResult: TypeAlias = list[list[float]]


@dataclass(frozen=True)
class ImageOptions:
    model: str | None = None
    size: str | None = None
    quality: Literal["standard", "hd"] | None = None
    count: int = 1
    style: Literal["vivid", "natural"] | None = None


@dataclass(frozen=True)
class GeneratedImage:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass(frozen=True)
class ImageResult:
    images: Sequence[GeneratedImage]


class AI(Protocol):
    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        ...

    def chat_stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> Iterator[ChatStreamEvent]:
        ...

    async def achat_stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        ...

    def transcribe(
        self,
        file: AudioFileInput,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        ...

    def embed(
        self,
        texts: Sequence[str],
        options: EmbedOptions | None = None,
    ) -> EmbedResult:
        ...

    def image(self, prompt: str, options: ImageOptions | None = None) -> ImageResult:
        ...
