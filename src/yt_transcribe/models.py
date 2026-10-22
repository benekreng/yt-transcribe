# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""YouTube-facing data models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from audio_transcribe import TranscriptResult

SourceType = Literal["video", "playlist", "channel", "manual"]
VideoStatus = Literal["transcribed", "cached", "failed"]


@dataclass(frozen=True)
class YouTubeVideoRef:
    video_id: str
    url: str
    title: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    duration_seconds: float | None = None
    published_at: datetime | None = None
    thumbnail_url: str | None = None
    source_type: SourceType | None = None
    source_id: str | None = None
    position: int | None = None
    is_short: bool = False
    is_live: bool = False
    # Spoken language as reported by YouTube, e.g. "en-US" or "de-DE".
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.published_at is not None:
            data["published_at"] = self.published_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> YouTubeVideoRef:
        published_at = data.get("published_at")
        return cls(
            video_id=str(data["video_id"]),
            url=str(data["url"]),
            title=data.get("title"),
            channel_id=data.get("channel_id"),
            channel_title=data.get("channel_title"),
            duration_seconds=(
                float(data["duration_seconds"])
                if data.get("duration_seconds") is not None
                else None
            ),
            published_at=(
                datetime.fromisoformat(published_at) if published_at else None
            ),
            thumbnail_url=data.get("thumbnail_url"),
            source_type=data.get("source_type"),
            source_id=data.get("source_id"),
            position=data.get("position"),
            is_short=bool(data.get("is_short", False)),
            is_live=bool(data.get("is_live", False)),
            language=data.get("language"),
        )


@dataclass(frozen=True)
class ResolvedSource:
    source_type: SourceType
    source_url: str | None = None
    source_id: str | None = None
    title: str | None = None
    video: YouTubeVideoRef | None = None
    videos: tuple[YouTubeVideoRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VideoTranscriptResult:
    video: YouTubeVideoRef
    status: VideoStatus
    transcript: TranscriptResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "status": self.status,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class MultiTranscriptResult:
    source_type: Literal["playlist", "channel", "manual"]
    source_url: str | None
    title: str | None
    videos: tuple[VideoTranscriptResult, ...]

    def successes(self) -> tuple[VideoTranscriptResult, ...]:
        return tuple(video for video in self.videos if video.status != "failed")

    def failures(self) -> tuple[VideoTranscriptResult, ...]:
        return tuple(video for video in self.videos if video.status == "failed")

    def combined_text(self) -> str:
        return "\n\n".join(
            item.transcript.text
            for item in self.videos
            if item.transcript is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "title": self.title,
            "videos": [video.to_dict() for video in self.videos],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "result.json").write_text(self.to_json(), encoding="utf-8")
        (path / "combined.txt").write_text(self.combined_text(), encoding="utf-8")
