# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from yt_transcribe import extract_video_id, resolve_source
from yt_transcribe.errors import InvalidSourceError


def test_extract_video_id_from_common_youtube_urls():
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_ambiguous_watch_playlist_defaults_to_video_unless_collection_preferred():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123"

    assert resolve_source(url).source_type == "video"
    playlist = resolve_source(url, prefer_collection=True)

    assert playlist.source_type == "playlist"
    assert playlist.source_id == "PL123"


def test_manual_lists_reject_playlist_urls():
    with pytest.raises(InvalidSourceError):
        resolve_source(["https://www.youtube.com/playlist?list=PL123"])
