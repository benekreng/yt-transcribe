# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from yt_transcribe.models import YouTubeVideoRef
from yt_transcribe.youtube_client import _video_from_ytdlp


def test_video_language_is_read_from_ytdlp_and_round_trips():
    video = _video_from_ytdlp(
        {"id": "78GTFUAQGDw", "title": "SpaceX - Tesla Fusion", "language": "de-DE", "duration": 1530},
        source_type="video",
    )

    assert video.language == "de-DE"
    assert YouTubeVideoRef.from_dict(video.to_dict()).language == "de-DE"
