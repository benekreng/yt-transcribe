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


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def transcribe_chunk(self, chunk: AudioChunk, options: AudioTranscribeOptions) -> ChunkTranscript:
        return ChunkTranscript(
            chunk=chunk,
            text=f"chunk {chunk.index}",
            segments=(
                TranscriptSegment(
                    text=f"chunk {chunk.index}",
                    start=chunk.nominal_start,
                    end=chunk.nominal_end,
                ),
            ),
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available")
def test_audio_transcriber_chunks_generated_wav_with_real_ffmpeg(tmp_path):
    source = tmp_path / "source.wav"
    _write_sine_wav(source, duration_seconds=1.2)

    transcriber = AudioTranscriber(backend=FakeBackend())
    result = transcriber.transcribe(
        source,
        AudioTranscribeOptions(
            source_id="integration:wav",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "work",
            chunking=ChunkingOptions(
                max_duration_seconds=0.5,
                overlap_seconds=0.05,
                output_format="mp3",
            ),
        ),
    )

    assert result.source_id == "integration:wav"
    assert result.text == "chunk 0\n\nchunk 1\n\nchunk 2"
    assert len(result.chunks) == 3
    assert [segment.text for segment in result.segments] == ["chunk 0", "chunk 1", "chunk 2"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available")
def test_audio_transcriber_reports_chunk_progress(tmp_path):
    source = tmp_path / "source.wav"
    _write_sine_wav(source, duration_seconds=1.2)
    events = []
    transcriber = AudioTranscriber(backend=FakeBackend())

    transcriber.transcribe(
        source,
        AudioTranscribeOptions(
            source_id="integration:progress",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "work",
            progress=events.append,
            chunking=ChunkingOptions(
                max_duration_seconds=0.5,
                overlap_seconds=0.05,
            ),
        ),
    )

    names = [event.name for event in events]
    assert names[0:2] == ["chunking.start", "chunking.complete"]
    assert names.count("chunk.start") == 3
    assert names.count("chunk.complete") == 3
    assert names[-1] == "transcription.complete"


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available")
def test_parallel_chunker_encodes_opus_chunks_in_order(tmp_path):
    from audio_transcribe import FFmpegChunker

    source = tmp_path / "source.wav"
    _write_sine_wav(source, duration_seconds=2.2)

    chunks = FFmpegChunker(concurrency=4).chunk(
        source,
        tmp_path / "chunks",
        ChunkingOptions(max_duration_seconds=0.5, overlap_seconds=0.05, output_format="webm", bitrate="32k"),
    )

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3, 4]
    assert [chunk.nominal_start for chunk in chunks] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert all(Path(chunk.file_path).suffix == ".webm" and chunk.size_bytes > 0 for chunk in chunks)
