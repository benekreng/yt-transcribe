# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio transcription client for OpenAI-compatible transcription endpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ai_providers.clients.http import OpenAIHTTPClient
from ai_providers.config.providers import OPENAI_CONFIG
from ai_providers.models import (
    AudioFileInput,
    TranscribeOptions,
    TranscribeResult,
    TranscriptionSegment,
    TranscriptionToken,
    TranscriptionWord,
)


class TranscriptionClient:
    def __init__(self, http: OpenAIHTTPClient) -> None:
        self._http = http

    def transcribe(
        self,
        file: AudioFileInput,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        options = options or TranscribeOptions()
        response_format = (
            options.response_format
            or str(OPENAI_CONFIG["transcribe"]["default_response_format"])
        )
        model = options.model or str(OPENAI_CONFIG["transcribe"]["model"])
        fields: dict[str, str | list[str]] = {
            "model": model,
            "response_format": response_format,
        }
        if options.language:
            # gpt-transcribe takes a `languages` list instead of `language` and rejects
            # both together; older models (gpt-4o-transcribe, whisper) reject `languages`.
            if model.startswith("gpt-transcribe"):
                fields["languages[]"] = [options.language]
            else:
                fields["language"] = options.language
        if options.prompt:
            fields["prompt"] = options.prompt
        if options.temperature is not None:
            fields["temperature"] = str(options.temperature)
        if options.timestamp_granularities:
            fields["timestamp_granularities[]"] = list(options.timestamp_granularities)

        _, raw_body = self._http.post_multipart(
            "/audio/transcriptions",
            fields=fields,
            file=file,
        )

        if response_format in {"text", "srt", "vtt"}:
            return TranscribeResult(text=raw_body.decode("utf-8"))

        data = json.loads(raw_body.decode("utf-8"))
        return _parse_transcribe_result(data)


def _parse_transcribe_result(data: Mapping[str, Any]) -> TranscribeResult:
    return TranscribeResult(
        text=str(data.get("text", "")),
        language=_parse_language(data),
        duration=float(data["duration"]) if data.get("duration") is not None else None,
        segments=tuple(_parse_segment(segment) for segment in data.get("segments") or []),
        words=tuple(
            TranscriptionWord(
                text=str(word.get("word", word.get("text", ""))),
                start=float(word.get("start", 0)),
                end=float(word.get("end", 0)),
            )
            for word in data.get("words") or []
            if isinstance(word, Mapping)
        ),
    )


def _parse_language(data: Mapping[str, Any]) -> str | None:
    if data.get("language"):
        return str(data["language"])
    # Newer models (e.g. gpt-transcribe) report a list: {"languages": [{"code": "en"}]}.
    languages = data.get("languages")
    if isinstance(languages, list) and languages and isinstance(languages[0], Mapping):
        code = languages[0].get("code")
        return str(code) if code else None
    return None


def _parse_segment(data: Mapping[str, Any]) -> TranscriptionSegment:
    return TranscriptionSegment(
        text=str(data.get("text", "")),
        start=float(data.get("start", 0)),
        end=float(data.get("end", 0)),
        tokens=tuple(
            TranscriptionToken(
                text=str(token.get("text", "")),
                start=float(token.get("start", 0)),
                end=float(token.get("end", 0)),
            )
            for token in data.get("tokens", [])
            if isinstance(token, Mapping)
        ),
        avg_logprob=_optional_float(data.get("avg_logprob")),
        compression_ratio=_optional_float(data.get("compression_ratio")),
        no_speech_prob=_optional_float(data.get("no_speech_prob")),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
