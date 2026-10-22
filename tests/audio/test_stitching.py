# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from audio_transcribe.models import AudioChunk, ChunkTranscript, TranscriptSegment
from audio_transcribe.stitching import merge_chunk_transcripts


def test_merge_chunk_transcripts_drops_overlap_segments_by_midpoint():
    chunk_0 = AudioChunk(
        index=0,
        file_path="0.mp3",
        start=0,
        end=12,
        nominal_start=0,
        nominal_end=10,
        size_bytes=1,
        format="mp3",
    )
    chunk_1 = AudioChunk(
        index=1,
        file_path="1.mp3",
        start=8,
        end=20,
        nominal_start=10,
        nominal_end=20,
        size_bytes=1,
        format="mp3",
    )

    _, segments = merge_chunk_transcripts(
        [
            ChunkTranscript(
                chunk=chunk_0,
                text="a",
                segments=(
                    TranscriptSegment("kept", 1, 2),
                    TranscriptSegment("overlap", 10.5, 11.5),
                ),
            ),
            ChunkTranscript(
                chunk=chunk_1,
                text="b",
                segments=(
                    TranscriptSegment("kept two", 10.5, 11.5),
                    TranscriptSegment("kept three", 18, 19),
                ),
            ),
        ]
    )

    assert [segment.text for segment in segments] == ["kept", "kept two", "kept three"]
