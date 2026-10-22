# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Options for audio transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audio_transcribe.progress import ProgressCallback


@dataclass(frozen=True)
class ChunkingOptions:
    max_duration_seconds: float = 600
    max_size_mb: float = 24
    overlap_seconds: float = 2
    silence_aware: bool = False
    output_format: str = "mp3"
    sample_rate: int = 16_000
    mono: bool = True
    # Encoder bitrate such as "32k". None keeps the codec default (64k for lossy formats).
    bitrate: str | None = None


@dataclass(frozen=True)
class AudioTranscribeOptions:
    language: str | None = None
    prompt: str | None = None
    model: str | None = None
    source_id: str | None = None
    provider_name: str = "openai"
    skip_existing: bool = True
    force: bool = False
    keep_chunks: bool = False
    concurrency: int = 3
    cache_dir: Path | str | None = None
    output_dir: Path | str | None = None
    chunking: ChunkingOptions = field(default_factory=ChunkingOptions)
    progress: ProgressCallback | None = None
