# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transcription backend interface."""

from __future__ import annotations

from typing import Protocol

from audio_transcribe.models import AudioChunk, ChunkTranscript
from audio_transcribe.options import AudioTranscribeOptions


class TranscriptionBackend(Protocol):
    name: str
    model: str

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        options: AudioTranscribeOptions,
    ) -> ChunkTranscript:
        ...
