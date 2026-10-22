# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transcript stitching and overlap dedupe."""

from __future__ import annotations

from audio_transcribe.exporters import segment_midpoint
from audio_transcribe.models import ChunkTranscript, TranscriptSegment


def merge_chunk_transcripts(
    chunks: tuple[ChunkTranscript, ...] | list[ChunkTranscript],
) -> tuple[str, tuple[TranscriptSegment, ...]]:
    ordered = sorted(chunks, key=lambda item: item.chunk.index)
    segments: list[TranscriptSegment] = []
    text_parts: list[str] = []

    for chunk_transcript in ordered:
        text = chunk_transcript.text.strip()
        if text:
            text_parts.append(text)

        for segment in chunk_transcript.segments:
            midpoint = segment_midpoint(segment)
            if (
                chunk_transcript.chunk.nominal_start
                <= midpoint
                <= chunk_transcript.chunk.nominal_end
            ):
                segments.append(segment)

    return "\n\n".join(text_parts), tuple(sorted(segments, key=lambda item: item.start))
