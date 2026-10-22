# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Options for YouTube transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from audio_transcribe import ChunkingOptions, ProgressCallback


@dataclass(frozen=True)
class ListVideosOptions:
    max_items: int | None = None
    include_shorts: bool = False
    include_livestreams: bool = False
    order: str = "source"
    since: datetime | None = None
    until: datetime | None = None


@dataclass(frozen=True)
class YTTranscribeOptions:
    language: str | None = None
    prompt: str | None = None
    model: str | None = None
    skip_existing: bool = True
    force: bool = False
    continue_on_error: bool = True
    concurrency: int = 3
    cache_dir: Path | str | None = None
    output_dir: Path | str | None = None
    keep_audio: bool = False
    chunking: ChunkingOptions = field(default_factory=ChunkingOptions)
    listing: ListVideosOptions = field(default_factory=ListVideosOptions)
    progress: ProgressCallback | None = None
