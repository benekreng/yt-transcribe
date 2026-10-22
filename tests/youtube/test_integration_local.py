# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
import shutil
import wave
from pathlib import Path

import pytest

from audio_transcribe import AudioTranscribeOptions, AudioTranscriber, ChunkingOptions
from audio_transcribe.models import AudioChunk, ChunkTranscript, TranscriptSegment
from yt_transcribe import YTTranscribeOptions, YTTranscriber, YouTubeVideoRef


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def transcribe_chunk(self, chunk: AudioChunk, options: AudioTranscribeOptions) -> ChunkTranscript:
        return ChunkTranscript(
            chunk=chunk,
            text=f"video chunk {chunk.index}",
            segments=(
                TranscriptSegment(
                    text=f"video chunk {chunk.index}",
                    start=chunk.nominal_start,
                    end=chunk.nominal_end,
                ),
            ),
        )


class FakeYouTube:
    def get_video(self, url_or_id):
        return YouTubeVideoRef(
            video_id="abc123abc12",
            url="https://www.youtube.com/watch?v=abc123abc12",
            title="Generated Audio",
            duration_seconds=1.0,
        )

    def list_videos(self, source_url, options=None):
        return (self.get_video("abc123abc12"),)


class LocalAudioDownloader:
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path

    def download_audio(self, video, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / f"{video.video_id}.wav"
        target.write_bytes(self.audio_path.read_bytes())
        return target


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available")
def test_yt_transcriber_uses_real_audio_engine_with_local_downloader(tmp_path):
    source = tmp_path / "source.wav"
    _write_sine_wav(source, duration_seconds=1.0)
    audio = AudioTranscriber(backend=FakeBackend())
    yt = YTTranscriber(
        audio=audio,
        youtube=FakeYouTube(),
        downloader=LocalAudioDownloader(source),
    )

    result = yt.transcribe(
        "https://www.youtube.com/watch?v=abc123abc12",
        YTTranscribeOptions(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            chunking=ChunkingOptions(max_duration_seconds=0.5, overlap_seconds=0.05),
        ),
    )

    assert result.status == "transcribed"
    assert result.video.title == "Generated Audio"
    assert result.transcript is not None
    assert result.transcript.text == "video chunk 0\n\nvideo chunk 1"
    assert (tmp_path / "out" / "generated-audio" / "transcript.json").exists()


def _write_sine_wav(path: Path, *, duration_seconds: float, sample_rate: int = 16_000) -> None:
    amplitude = 8000
    frequency = 440
    sample_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(sample_count):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
