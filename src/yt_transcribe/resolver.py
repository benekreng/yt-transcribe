# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Resolve user inputs into video or collection sources."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import parse_qs, urlparse

from yt_transcribe.errors import InvalidSourceError
from yt_transcribe.models import ResolvedSource, YouTubeVideoRef

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def resolve_source(source, *, prefer_collection: bool = False) -> ResolvedSource:
    if isinstance(source, YouTubeVideoRef):
        return ResolvedSource(source_type="video", video=source)

    if isinstance(source, str):
        playlist_id = extract_playlist_id(source)
        video_id = extract_video_id(source)
        if prefer_collection and playlist_id:
            return ResolvedSource(
                source_type="playlist",
                source_url=source,
                source_id=playlist_id,
            )
        if video_id:
            return ResolvedSource(
                source_type="video",
                video=YouTubeVideoRef(video_id=video_id, url=canonical_video_url(video_id)),
            )
        if playlist_id:
            return ResolvedSource(
                source_type="playlist",
                source_url=source,
                source_id=playlist_id,
            )
        if is_channel_url(source):
            return ResolvedSource(source_type="channel", source_url=source)
        raise InvalidSourceError(f"Unsupported YouTube source: {source}")

    if isinstance(source, Sequence):
        videos = tuple(normalize_video_input(item) for item in source)
        return ResolvedSource(source_type="manual", videos=videos)

    raise InvalidSourceError(f"Unsupported source type: {type(source).__name__}")


def normalize_video_input(item: str | YouTubeVideoRef) -> YouTubeVideoRef:
    if isinstance(item, YouTubeVideoRef):
        return item
    if not isinstance(item, str):
        raise InvalidSourceError(f"Expected video URL or YouTubeVideoRef, got {type(item).__name__}")
    if extract_playlist_id(item) and not extract_video_id(item):
        raise InvalidSourceError("Lists passed to transcribe_many may only contain video URLs or refs.")
    if is_channel_url(item):
        raise InvalidSourceError("Lists passed to transcribe_many may only contain video URLs or refs.")
    video_id = extract_video_id(item)
    if not video_id:
        raise InvalidSourceError(f"Could not extract YouTube video ID from: {item}")
    return YouTubeVideoRef(video_id=video_id, url=canonical_video_url(video_id))


def extract_video_id(value: str) -> str | None:
    if VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if "v" in query and query["v"]:
        candidate = query["v"][0]
        return candidate if VIDEO_ID_RE.match(candidate) else None

    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/v/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def extract_playlist_id(value: str) -> str | None:
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if "list" not in query or not query["list"]:
        return None
    return query["list"][0]


def is_channel_url(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    return (
        path.startswith("/@")
        or path.startswith("/channel/")
        or path.startswith("/c/")
        or path.startswith("/user/")
    )


def canonical_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
