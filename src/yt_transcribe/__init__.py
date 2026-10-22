# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""YouTube-first transcription API."""

from audio_transcribe import ProgressCallback, ProgressEvent

from yt_transcribe.models import (
    MultiTranscriptResult,
    VideoTranscriptResult,
    YouTubeVideoRef,
)
from yt_transcribe.options import ListVideosOptions, YTTranscribeOptions
from yt_transcribe.resolver import (
    canonical_video_url,
    extract_playlist_id,
    extract_video_id,
    normalize_video_input,
    resolve_source,
)
from yt_transcribe.transcriber import YTTranscriber

__all__ = [
    "ListVideosOptions",
    "MultiTranscriptResult",
    "ProgressCallback",
    "ProgressEvent",
    "VideoTranscriptResult",
    "YTTranscribeOptions",
    "YTTranscriber",
    "YouTubeVideoRef",
    "canonical_video_url",
    "extract_playlist_id",
    "extract_video_id",
    "normalize_video_input",
    "resolve_source",
]
