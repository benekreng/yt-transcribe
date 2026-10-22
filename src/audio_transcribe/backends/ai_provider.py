# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Backend adapter around the ai_providers facade."""

from __future__ import annotations

from ai_providers import AI, TranscribeOptions

from audio_transcribe.errors import BackendTranscriptionError
from audio_transcribe.models import AudioChunk, ChunkTranscript, TranscriptSegment
from audio_transcribe.options import AudioTranscribeOptions


class AIProviderTranscriptionBackend:
    name = "openai"

    def __init__(self, ai: AI, *, model: str = "gpt-transcribe") -> None:
        self.ai = ai
        self.model = model

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        options: AudioTranscribeOptions,
    ) -> ChunkTranscript:
        try:
            result = self.ai.transcribe(
                chunk.file_path,
                TranscribeOptions(
                    model=options.model or self.model,
                    language=options.language,
                    prompt=options.prompt,
                    response_format="json",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - preserve backend cause
            raise BackendTranscriptionError(
                f"Transcription failed for chunk {chunk.index}."
            ) from exc

        segments = tuple(
            TranscriptSegment(
                text=segment.text,
                start=chunk.start + segment.start,
                end=chunk.start + segment.end,
            )
            for segment in result.segments
        )
        if not segments:
            segments = (
                TranscriptSegment(
                    text=result.text,
                    start=chunk.nominal_start,
                    end=chunk.nominal_end,
                ),
            )

        return ChunkTranscript(chunk=chunk, text=result.text, segments=segments)
