# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

from audio_transcribe import TranscriptResult
from yt_transcribe import YTTranscribeOptions, YTTranscriber, YouTubeVideoRef


def _transcript(source_id="youtube:abc123"):
    return TranscriptResult(
        text="hello",
        segments=tuple(),
        language="en",
        duration_seconds=12,
        chunks=tuple(),
        source_id=source_id,
        backend="fake",
        model="fake-model",
    )


class FakeBackend:
    name = "fake"
    model = "fake-model"


class FakeCache:
    def __init__(self, result=None):
        self.result = result

    def key_for(self, **kwargs):
        return kwargs["source_id"]

    def load_result(self, key):
        return self.result


class FakeAudio:
    def __init__(self, cached=None):
        self.backend = FakeBackend()
        self.cache = FakeCache(cached)
        self.calls = []

    def transcribe(self, source, options):
        self.calls.append((source, options))
        return _transcript(options.source_id)


class FakeYouTube:
    def get_video(self, url_or_id):
        return YouTubeVideoRef(
            video_id="abc123",
            url="https://www.youtube.com/watch?v=abc123abc12",
            title="Example",
            duration_seconds=12,
        )

    def list_videos(self, source_url, options=None):
        return (
            YouTubeVideoRef(
                video_id="abc123abc12",
                url="https://www.youtube.com/watch?v=abc123abc12",
                title="One",
            ),
            YouTubeVideoRef(
                video_id="def456def45",
                url="https://www.youtube.com/watch?v=def456def45",
                title="Two",
            ),
        )


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def download_audio(self, video, output_dir):
        self.calls.append(video.video_id)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        audio = path / f"{video.video_id}.mp3"
        audio.write_bytes(b"audio")
        return audio


def test_transcribe_video_skips_download_when_cached():
    cached = _transcript()
    audio = FakeAudio(cached=cached)
    downloader = FakeDownloader()
    yt = YTTranscriber(audio=audio, youtube=FakeYouTube(), downloader=downloader)

    result = yt.transcribe(YouTubeVideoRef(video_id="abc123abc12", url="https://youtu.be/abc123abc12"))

    assert result.status == "cached"
    assert result.transcript is cached
    assert downloader.calls == []
    assert audio.calls == []


def test_transcriber_reports_metadata_and_cache_progress():
    events = []
    audio = FakeAudio(cached=_transcript())
    yt = YTTranscriber(
        audio=audio,
        youtube=FakeYouTube(),
        downloader=FakeDownloader(),
    )

    yt.transcribe(
        "https://www.youtube.com/watch?v=abc123abc12",
        YTTranscribeOptions(progress=events.append),
    )

    assert [event.name for event in events] == [
        "source.resolve",
        "metadata.start",
        "metadata.complete",
        "cache.hit",
    ]
    assert events[2].details["title"] == "Example"


def test_multi_transcribe_returns_batch_result_for_channel_source():
    audio = FakeAudio()
    downloader = FakeDownloader()
    yt = YTTranscriber(audio=audio, youtube=FakeYouTube(), downloader=downloader)

    result = yt.multi_transcribe("https://www.youtube.com/@example")

    assert result.source_type == "channel"
    assert len(result.videos) == 2
    assert [item.status for item in result.videos] == ["transcribed", "transcribed"]
    assert downloader.calls == ["abc123abc12", "def456def45"]
