# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio transcription errors."""

from __future__ import annotations


class AudioTranscribeError(RuntimeError):
    """Base error for audio transcription."""


class FFmpegNotFoundError(AudioTranscribeError):
    """ffmpeg or ffprobe is not available on PATH."""


class AudioPreparationError(AudioTranscribeError):
    """Audio probing or preparation failed."""


class ChunkingError(AudioTranscribeError):
    """Audio chunking failed."""


class BackendTranscriptionError(AudioTranscribeError):
    """The transcription backend failed."""


class CacheError(AudioTranscribeError):
    """Cache read/write failed."""
