# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Source-agnostic audio transcription engine."""

from audio_transcribe.alignment import (
    AlignedToken,
    Alignment,
    TimedWord,
    align_text_to_words,
    normalize_token,
    split_sentences,
)
from audio_transcribe.cache import TranscriptCache
from audio_transcribe.chunking import FFmpegChunker, ChunkWindow, plan_chunk_windows
from audio_transcribe.models import (
    AudioChunk,
    ChunkTranscript,
    PreparedAudio,
    TranscriptResult,
    TranscriptSegment,
)
from audio_transcribe.options import AudioTranscribeOptions, ChunkingOptions
from audio_transcribe.progress import ProgressCallback, ProgressEvent
from audio_transcribe.transcriber import AudioTranscriber

__all__ = [
    "AlignedToken",
    "Alignment",
    "AudioChunk",
    "AudioTranscribeOptions",
    "AudioTranscriber",
    "ChunkTranscript",
    "ChunkWindow",
    "ChunkingOptions",
    "FFmpegChunker",
    "PreparedAudio",
    "ProgressCallback",
    "ProgressEvent",
    "TimedWord",
    "TranscriptCache",
    "TranscriptResult",
    "TranscriptSegment",
    "align_text_to_words",
    "normalize_token",
    "plan_chunk_windows",
    "split_sentences",
]
