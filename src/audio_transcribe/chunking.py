# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ffmpeg-backed audio chunking."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from audio_transcribe.errors import ChunkingError
from audio_transcribe.models import AudioChunk
from audio_transcribe.options import ChunkingOptions
from audio_transcribe.preparation import ensure_binary, probe_audio


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: float
    end: float
    nominal_start: float
    nominal_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_chunk_windows(duration_seconds: float, options: ChunkingOptions) -> tuple[ChunkWindow, ...]:
    if duration_seconds <= 0:
        return tuple()
    if options.max_duration_seconds <= 0:
        raise ChunkingError("max_duration_seconds must be greater than zero.")
    if options.overlap_seconds < 0:
        raise ChunkingError("overlap_seconds cannot be negative.")

    windows: list[ChunkWindow] = []
    nominal_start = 0.0
    index = 0
    while nominal_start < duration_seconds:
        nominal_end = min(duration_seconds, nominal_start + options.max_duration_seconds)
        start = max(0.0, nominal_start - options.overlap_seconds)
        end = min(duration_seconds, nominal_end + options.overlap_seconds)
        windows.append(
            ChunkWindow(
                index=index,
                start=start,
                end=end,
                nominal_start=nominal_start,
                nominal_end=nominal_end,
            )
        )
        nominal_start = nominal_end
        index += 1
    return tuple(windows)


class FFmpegChunker:
    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        concurrency: int = 1,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        # Number of ffmpeg processes encoding chunks at the same time.
        self.concurrency = max(1, concurrency)

    def chunk(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        options: ChunkingOptions,
    ) -> tuple[AudioChunk, ...]:
        ensure_binary(self.ffmpeg_path)
        prepared = probe_audio(input_file, ffprobe_path=self.ffprobe_path)
        windows = plan_chunk_windows(prepared.duration_seconds, options)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def encode(window: ChunkWindow) -> AudioChunk:
            return self._encode_window(Path(input_file), output_path, window, options)

        if self.concurrency == 1 or len(windows) <= 1:
            return tuple(encode(window) for window in windows)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return tuple(executor.map(encode, windows))

    def _encode_window(
        self,
        input_file: Path,
        output_path: Path,
        window: ChunkWindow,
        options: ChunkingOptions,
    ) -> AudioChunk:
        chunk_path = output_path / f"chunk_{window.index:04d}.{_extension(options.output_format)}"
        args = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            f"{window.start:.3f}",
            "-t",
            f"{window.duration:.3f}",
            "-i",
            str(input_file),
            "-vn",
            *_audio_shape_args(options),
            *_codec_args(options.output_format, options.bitrate),
            str(chunk_path),
        ]
        try:
            subprocess.run(args, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise ChunkingError(exc.stderr.strip() or "ffmpeg failed") from exc

        size_bytes = chunk_path.stat().st_size
        max_bytes = int(options.max_size_mb * 1024 * 1024)
        if size_bytes > max_bytes:
            raise ChunkingError(
                f"Chunk {window.index} is {size_bytes} bytes, exceeding {max_bytes} bytes."
            )

        return AudioChunk(
            index=window.index,
            file_path=str(chunk_path),
            start=window.start,
            end=window.end,
            nominal_start=window.nominal_start,
            nominal_end=window.nominal_end,
            size_bytes=size_bytes,
            format=_extension(options.output_format),
        )


def _audio_shape_args(options: ChunkingOptions) -> list[str]:
    args: list[str] = []
    if options.mono:
        args.extend(["-ac", "1"])
    if options.sample_rate:
        args.extend(["-ar", str(options.sample_rate)])
    return args


def _codec_args(format_name: str, bitrate: str | None = None) -> list[str]:
    rate = bitrate or "64k"
    match format_name.lower():
        case "mp3" | "mpeg" | "mpga":
            return ["-acodec", "libmp3lame", "-b:a", rate]
        case "wav":
            return ["-acodec", "pcm_s16le"]
        case "flac":
            return ["-acodec", "flac", "-compression_level", "5"]
        case "m4a":
            return ["-acodec", "aac", "-b:a", rate]
        case "ogg":
            return ["-acodec", "libvorbis", "-b:a", rate]
        case "webm":
            return ["-acodec", "libopus", "-b:a", rate]
        case _:
            raise ChunkingError(f"Unsupported chunk format: {format_name}")


def _extension(format_name: str) -> str:
    lowered = format_name.lower()
    if lowered in {"mpeg", "mpga"}:
        return "mp3"
    return lowered
