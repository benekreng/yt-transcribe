# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Public provider facade."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from ai_providers.clients.chat import ChatClient
from ai_providers.clients.embed import EmbedClient
from ai_providers.clients.http import OpenAIHTTPClient
from ai_providers.clients.image import ImageClient
from ai_providers.clients.transcribe import TranscriptionClient
from ai_providers.config.providers import OPENAI_CONFIG
from ai_providers.errors import ProviderConfigurationError
from ai_providers.models import (
    AudioFileInput,
    ChatMessage,
    ChatOptions,
    ChatResult,
    ChatStreamEvent,
    EmbedOptions,
    EmbedResult,
    ImageOptions,
    ImageResult,
    OpenAIOptions,
    TranscribeOptions,
    TranscribeResult,
)


class AIFacade:
    """Single entry point for provider interactions."""

    def __init__(self, *, openai: OpenAIOptions) -> None:
        if not openai.api_key:
            raise ProviderConfigurationError("OpenAI api_key is required.")

        http = OpenAIHTTPClient(
            api_key=openai.api_key,
            base_url=openai.base_url or str(OPENAI_CONFIG["base_url"]),
            organization=openai.organization,
            project=openai.project,
        )
        self._chat = ChatClient(http)
        self._transcribe = TranscriptionClient(http)
        self._embed = EmbedClient(http)
        self._image = ImageClient(http)

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        return self._chat.chat(messages, options)

    def chat_stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> Iterator[ChatStreamEvent]:
        return self._chat.chat_stream(messages, options)

    async def achat_stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        for event in self.chat_stream(messages, options):
            yield event

    def transcribe(
        self,
        file: AudioFileInput,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        return self._transcribe.transcribe(file, options)

    def embed(
        self,
        texts: Sequence[str],
        options: EmbedOptions | None = None,
    ) -> EmbedResult:
        return self._embed.embed(tuple(texts), options)

    def image(self, prompt: str, options: ImageOptions | None = None) -> ImageResult:
        return self._image.image(prompt, options)


def create_ai(*, openai: OpenAIOptions | Mapping[str, Any]) -> AIFacade:
    """Create the provider facade.

    This function reads no environment variables. Applications decide where
    credentials come from and inject them here.
    """

    openai_options = (
        openai if isinstance(openai, OpenAIOptions) else OpenAIOptions(**dict(openai))
    )
    return AIFacade(openai=openai_options)
