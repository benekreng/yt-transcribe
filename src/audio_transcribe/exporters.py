# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transcript export helpers."""

from __future__ import annotations

from audio_transcribe.models import TranscriptResult, TranscriptSegment


def to_text(result: TranscriptResult, *, include_timestamps: bool = True) -> str:
    if not include_timestamps or not result.segments:
        return result.text
    return "\n".join(
        f"[{format_timestamp(segment.start, sep=':')}] {segment.text}"
        for segment in result.segments
    )


def to_srt(result: TranscriptResult) -> str:
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(segment.start, sep=',')} --> {format_timestamp(segment.end, sep=',')}",
                    segment.text.strip(),
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(result: TranscriptResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        lines.extend(
            [
                f"{format_timestamp(segment.start, sep='.')} --> {format_timestamp(segment.end, sep='.')}",
                segment.text.strip(),
                "",
            ]
        )
    return "\n".join(lines)


def format_timestamp(seconds: float, *, sep: str) -> str:
    milliseconds = round((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{milliseconds:03d}"


def segment_midpoint(segment: TranscriptSegment) -> float:
    return segment.start + ((segment.end - segment.start) / 2)
