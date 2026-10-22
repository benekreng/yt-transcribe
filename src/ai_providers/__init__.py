# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unified provider facade for model calls."""

from ai_providers.config.providers import PROVIDER_CONFIG
from ai_providers.facade import AIFacade, create_ai
from ai_providers.models import (
    AI,
    AudioFileInput,
    ChatMessage,
    ChatMessageContent,
    ChatOptions,
    ChatResult,
    ChatStreamEvent,
    CreateAIOptions,
    EmbedOptions,
    EmbedResult,
    GeneratedImage,
    ImageOptions,
    ImageResult,
    OpenAIOptions,
    ResponseFormat,
    TokenUsage,
    TranscribeOptions,
    TranscribeResult,
    TimestampGranularity,
    TranscriptionSegment,
    TranscriptionToken,
    TranscriptionWord,
)

__all__ = [
    "AI",
    "AIFacade",
    "AudioFileInput",
    "ChatMessage",
    "ChatMessageContent",
    "ChatOptions",
    "ChatResult",
    "ChatStreamEvent",
    "CreateAIOptions",
    "EmbedOptions",
    "EmbedResult",
    "GeneratedImage",
    "ImageOptions",
    "ImageResult",
    "OpenAIOptions",
    "PROVIDER_CONFIG",
    "ResponseFormat",
    "TokenUsage",
    "TranscribeOptions",
    "TranscribeResult",
    "TimestampGranularity",
    "TranscriptionSegment",
    "TranscriptionToken",
    "TranscriptionWord",
    "create_ai",
]
