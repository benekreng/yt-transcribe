# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""High-level audio transcription engine."""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from audio_transcribe.backends.base import TranscriptionBackend
from audio_transcribe.cache import TranscriptCache
from audio_transcribe.chunking import FFmpegChunker
from audio_transcribe.models import ChunkTranscript, TranscriptResult
from audio_transcribe.options import AudioTranscribeOptions
from audio_transcribe.preparation import hash_file, probe_audio
from audio_transcribe.progress import emit_progress
from audio_transcribe.stitching import merge_chunk_transcripts


class AudioTranscriber:
    def __init__(
        self,
        *,
        backend: TranscriptionBackend,
        chunker: FFmpegChunker | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        self.backend = backend
        self.chunker = chunker or FFmpegChunker()
        self.cache = cache or TranscriptCache()

    def transcribe(
        self,
        source: str | Path | bytes | bytearray,
        options: AudioTranscribeOptions | None = None,
    ) -> TranscriptResult:
        options = options or AudioTranscribeOptions()
        with tempfile.TemporaryDirectory(prefix="audio-transcribe-") as temp_dir:
            input_path = _materialize_source(source, Path(temp_dir))
            source_id = options.source_id or hash_file(input_path)
            model = options.model or self.backend.model
            cache = (
                self.cache
                if options.cache_dir is None
                else TranscriptCache(options.cache_dir)
            )
            cache_key = cache.key_for(
                source_id=source_id,
                backend=self.backend.name,
                model=model,
                options=options,
            )
            if options.skip_existing and not options.force:
                cached = cache.load_result(cache_key)
                if cached is not None:
                    emit_progress(
                        options.progress,
                        "cache.hit",
                        source_id=source_id,
                        level="transcript",
                    )
                    return cached

            work_dir = Path(options.output_dir or temp_dir) / "chunks"
            emit_progress(options.progress, "chunking.start", source_id=source_id)
            chunks = self.chunker.chunk(input_path, work_dir, options.chunking)
            emit_progress(
                options.progress,
                "chunking.complete",
                source_id=source_id,
                chunks=len(chunks),
            )
            chunk_transcripts = self._transcribe_chunks(
                chunks=chunks,
                cache=cache,
                cache_key=cache_key,
                options=options,
            )
            merged_text, merged_segments = merge_chunk_transcripts(chunk_transcripts)
            prepared = probe_audio(input_path)
            result = TranscriptResult(
                text=merged_text,
                segments=merged_segments,
                language=options.language,
                duration_seconds=prepared.duration_seconds,
                chunks=tuple(sorted(chunk_transcripts, key=lambda item: item.chunk.index)),
                source_id=source_id,
                backend=self.backend.name,
                model=model,
            )
            cache.save_result(cache_key, result)
            emit_progress(
                options.progress,
                "transcription.complete",
                source_id=source_id,
                chunks=len(chunk_transcripts),
                duration_seconds=prepared.duration_seconds,
            )

            if not options.keep_chunks and options.output_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)

            return result

    def transcribe_many(
        self,
        sources: list[str | Path] | tuple[str | Path, ...],
        options: AudioTranscribeOptions | None = None,
    ) -> tuple[TranscriptResult, ...]:
        return tuple(self.transcribe(source, options) for source in sources)

    def _transcribe_chunks(
        self,
        *,
        chunks,
        cache: TranscriptCache,
        cache_key: str,
        options: AudioTranscribeOptions,
    ) -> tuple[ChunkTranscript, ...]:
        results: dict[int, ChunkTranscript] = {}
        pending = []
        for chunk in chunks:
            cached = None if options.force else cache.load_chunk(cache_key, chunk.index)
            if cached is not None:
                results[chunk.index] = cached
                emit_progress(
                    options.progress,
                    "chunk.cache_hit",
                    index=chunk.index + 1,
                    total=len(chunks),
                )
            else:
                pending.append(chunk)

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as executor:
                future_to_chunk = {}
                for chunk in pending:
                    emit_progress(
                        options.progress,
                        "chunk.start",
                        index=chunk.index + 1,
                        total=len(chunks),
                    )
                    future = executor.submit(
                        self.backend.transcribe_chunk,
                        chunk,
                        options,
                    )
                    future_to_chunk[future] = chunk
                for future in as_completed(future_to_chunk):
                    transcript = future.result()
                    cache.save_chunk(cache_key, transcript)
                    results[transcript.chunk.index] = transcript
                    emit_progress(
                        options.progress,
                        "chunk.complete",
                        index=transcript.chunk.index + 1,
                        total=len(chunks),
                    )

        return tuple(results[index] for index in sorted(results))


def _materialize_source(source: str | Path | bytes | bytearray, temp_dir: Path) -> Path:
    if isinstance(source, str | Path):
        return Path(source)

    path = temp_dir / "input-audio"
    path.write_bytes(bytes(source))
    return path
