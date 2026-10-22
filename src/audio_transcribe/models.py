# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core audio transcription data models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedAudio:
    source_id: str
    file_path: str
    format: str | None
    duration_seconds: float
    size_bytes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparedAudio:
        return cls(
            source_id=str(data["source_id"]),
            file_path=str(data["file_path"]),
            format=data.get("format"),
            duration_seconds=float(data["duration_seconds"]),
            size_bytes=int(data["size_bytes"]),
        )


@dataclass(frozen=True)
class AudioChunk:
    index: int
    file_path: str
    start: float
    end: float
    nominal_start: float
    nominal_end: float
    size_bytes: int
    format: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioChunk:
        return cls(
            index=int(data["index"]),
            file_path=str(data["file_path"]),
            start=float(data["start"]),
            end=float(data["end"]),
            nominal_start=float(data["nominal_start"]),
            nominal_end=float(data["nominal_end"]),
            size_bytes=int(data["size_bytes"]),
            format=str(data["format"]),
        )


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        return cls(
            text=str(data["text"]),
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=(
                float(data["confidence"]) if data.get("confidence") is not None else None
            ),
            speaker=data.get("speaker"),
        )


@dataclass(frozen=True)
class ChunkTranscript:
    chunk: AudioChunk
    text: str
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkTranscript:
        return cls(
            chunk=AudioChunk.from_dict(data["chunk"]),
            text=str(data["text"]),
            segments=tuple(
                TranscriptSegment.from_dict(segment)
                for segment in data.get("segments", [])
            ),
        )


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: tuple[TranscriptSegment, ...]
    language: str | None
    duration_seconds: float
    chunks: tuple[ChunkTranscript, ...]
    source_id: str
    backend: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self, *, include_timestamps: bool = True) -> str:
        from audio_transcribe.exporters import to_text

        return to_text(self, include_timestamps=include_timestamps)

    def to_srt(self) -> str:
        from audio_transcribe.exporters import to_srt

        return to_srt(self)

    def to_vtt(self) -> str:
        from audio_transcribe.exporters import to_vtt

        return to_vtt(self)

    def save(self, output_dir: str | Path, *, base_name: str = "transcript") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / f"{base_name}.json").write_text(self.to_json(), encoding="utf-8")
        (output_path / f"{base_name}.txt").write_text(self.to_text(), encoding="utf-8")
        (output_path / f"{base_name}.srt").write_text(self.to_srt(), encoding="utf-8")
        (output_path / f"{base_name}.vtt").write_text(self.to_vtt(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptResult:
        return cls(
            text=str(data["text"]),
            segments=tuple(
                TranscriptSegment.from_dict(segment)
                for segment in data.get("segments", [])
            ),
            language=data.get("language"),
            duration_seconds=float(data["duration_seconds"]),
            chunks=tuple(
                ChunkTranscript.from_dict(chunk)
                for chunk in data.get("chunks", [])
            ),
            source_id=str(data["source_id"]),
            backend=str(data["backend"]),
            model=str(data["model"]),
        )
