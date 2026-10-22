# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""YouTube metadata/listing client."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Protocol

from yt_transcribe.errors import YouTubeClientError
from yt_transcribe.models import YouTubeVideoRef
from yt_transcribe.options import ListVideosOptions
from yt_transcribe.resolver import canonical_video_url, extract_video_id


class YouTubeClient(Protocol):
    def get_video(self, url_or_id: str) -> YouTubeVideoRef:
        ...

    def list_videos(self, source_url: str, options: ListVideosOptions | None = None) -> tuple[YouTubeVideoRef, ...]:
        ...


class YtDlpYouTubeClient:
    def __init__(self, *, yt_dlp_path: str = "yt-dlp") -> None:
        self.yt_dlp_path = yt_dlp_path

    def get_video(self, url_or_id: str) -> YouTubeVideoRef:
        url = canonical_video_url(url_or_id) if extract_video_id(url_or_id) == url_or_id else url_or_id
        data = self._run_json(["--dump-json", "--no-playlist", url])
        return _video_from_ytdlp(data, source_type="video")

    def list_videos(
        self,
        source_url: str,
        options: ListVideosOptions | None = None,
    ) -> tuple[YouTubeVideoRef, ...]:
        options = options or ListVideosOptions()
        args = ["--flat-playlist", "--dump-json", "--skip-download"]
        if options.max_items is not None:
            args.extend(["--playlist-end", str(options.max_items)])
        args.append(source_url)

        try:
            completed = subprocess.run(
                [self.yt_dlp_path, *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise YouTubeClientError(exc.stderr.strip() or "yt-dlp listing failed") from exc

        videos = []
        for index, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            video = _video_from_ytdlp(data, source_type=_source_type(source_url), position=index)
            if not options.include_shorts and video.is_short:
                continue
            if not options.include_livestreams and video.is_live:
                continue
            videos.append(video)
        return tuple(videos)

    def _run_json(self, args: list[str]) -> dict:
        try:
            completed = subprocess.run(
                [self.yt_dlp_path, *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise YouTubeClientError(exc.stderr.strip() or "yt-dlp failed") from exc
        return json.loads(completed.stdout)


def _video_from_ytdlp(
    data: dict,
    *,
    source_type: str,
    position: int | None = None,
) -> YouTubeVideoRef:
    video_id = data.get("id") or data.get("url")
    if not video_id:
        raise YouTubeClientError("yt-dlp returned an item without an id.")
    duration = data.get("duration")
    published_at = _parse_date(data.get("upload_date") or data.get("timestamp"))
    return YouTubeVideoRef(
        video_id=video_id,
        url=canonical_video_url(video_id),
        title=data.get("title"),
        channel_id=data.get("channel_id"),
        channel_title=data.get("channel") or data.get("uploader"),
        duration_seconds=float(duration) if duration is not None else None,
        published_at=published_at,
        thumbnail_url=data.get("thumbnail"),
        source_type=source_type,  # type: ignore[arg-type]
        source_id=data.get("playlist_id") or data.get("channel_id"),
        position=position or data.get("playlist_index"),
        is_short=bool(data.get("duration") is not None and data.get("duration") <= 60),
        is_live=bool(data.get("is_live") or data.get("live_status") == "is_live"),
        language=data.get("language"),
    )


def _source_type(source_url: str) -> str:
    return "playlist" if "list=" in source_url else "channel"


def _parse_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime(
            int(text[:4]),
            int(text[4:6]),
            int(text[6:8]),
            tzinfo=timezone.utc,
        )
    return None
