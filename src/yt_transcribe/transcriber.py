# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""High-level YouTube transcription API."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from audio_transcribe import AudioTranscribeOptions, AudioTranscriber, TranscriptCache
from audio_transcribe.progress import emit_progress

from yt_transcribe.cache import slugify
from yt_transcribe.downloader import AudioDownloader, YtDlpAudioDownloader
from yt_transcribe.models import (
    MultiTranscriptResult,
    ResolvedSource,
    VideoTranscriptResult,
    YouTubeVideoRef,
)
from yt_transcribe.options import ListVideosOptions, YTTranscribeOptions
from yt_transcribe.resolver import normalize_video_input, resolve_source
from yt_transcribe.youtube_client import YouTubeClient, YtDlpYouTubeClient


class YTTranscriber:
    def __init__(
        self,
        *,
        audio: AudioTranscriber,
        youtube: YouTubeClient | None = None,
        downloader: AudioDownloader | None = None,
    ) -> None:
        self.audio = audio
        self.youtube = youtube or YtDlpYouTubeClient()
        self.downloader = downloader or YtDlpAudioDownloader()

    def get_video(self, source: str | YouTubeVideoRef) -> YouTubeVideoRef:
        if isinstance(source, YouTubeVideoRef) and (
            source.title is not None or source.duration_seconds is not None
        ):
            return source
        fetched = self.youtube.get_video(source.video_id if isinstance(source, YouTubeVideoRef) else source)
        if not isinstance(source, YouTubeVideoRef):
            return fetched
        return replace(
            fetched,
            source_type=source.source_type or fetched.source_type,
            source_id=source.source_id or fetched.source_id,
            position=source.position if source.position is not None else fetched.position,
        )

    def list_videos(
        self,
        source,
        options: ListVideosOptions | None = None,
    ) -> tuple[YouTubeVideoRef, ...]:
        resolved = resolve_source(source, prefer_collection=True)
        if resolved.source_type == "video":
            return (self.get_video(resolved.video or source),)
        if resolved.source_type == "manual":
            return tuple(self.get_video(video) for video in resolved.videos)
        return self.youtube.list_videos(resolved.source_url or "", options)

    def transcribe(
        self,
        source,
        options: YTTranscribeOptions | None = None,
    ) -> VideoTranscriptResult | MultiTranscriptResult:
        options = options or YTTranscribeOptions()
        emit_progress(options.progress, "source.resolve", source=str(source))
        resolved = resolve_source(source)
        if resolved.source_type == "video":
            assert resolved.video is not None
            return self.transcribe_video(resolved.video, options)
        return self._transcribe_resolved_many(resolved, options)

    def multi_transcribe(
        self,
        source,
        options: YTTranscribeOptions | None = None,
    ) -> MultiTranscriptResult:
        options = options or YTTranscribeOptions()
        emit_progress(options.progress, "source.resolve", source=str(source))
        resolved = resolve_source(source, prefer_collection=True)
        if resolved.source_type == "video":
            assert resolved.video is not None
            resolved = ResolvedSource(source_type="manual", videos=(resolved.video,))
        return self._transcribe_resolved_many(resolved, options)

    def transcribe_many(
        self,
        videos: Sequence[str | YouTubeVideoRef],
        options: YTTranscribeOptions | None = None,
    ) -> MultiTranscriptResult:
        options = options or YTTranscribeOptions()
        emit_progress(options.progress, "source.resolve", sources=len(videos))
        resolved = ResolvedSource(
            source_type="manual",
            videos=tuple(normalize_video_input(video) for video in videos),
        )
        return self._transcribe_resolved_many(resolved, options)

    def transcribe_video(
        self,
        video: str | YouTubeVideoRef,
        options: YTTranscribeOptions | None = None,
    ) -> VideoTranscriptResult:
        options = options or YTTranscribeOptions()
        video_id = (
            video.video_id
            if isinstance(video, YouTubeVideoRef)
            else normalize_video_input(video).video_id
        )
        emit_progress(options.progress, "metadata.start", video_id=video_id)
        video_ref = self.get_video(video)
        emit_progress(
            options.progress,
            "metadata.complete",
            video_id=video_ref.video_id,
            title=video_ref.title,
            duration_seconds=video_ref.duration_seconds,
        )
        audio_options = _audio_options_for(video_ref, options)
        cached = _load_cached_audio_result(self.audio, audio_options)
        if cached is not None:
            emit_progress(
                options.progress,
                "cache.hit",
                video_id=video_ref.video_id,
                level="transcript",
            )
            return VideoTranscriptResult(
                video=video_ref,
                status="cached",
                transcript=cached,
            )

        with tempfile.TemporaryDirectory(prefix="yt-transcribe-") as temp_dir:
            download_dir = Path(options.output_dir or temp_dir) / "audio"
            emit_progress(
                options.progress,
                "download.start",
                video_id=video_ref.video_id,
            )
            audio_file = self.downloader.download_audio(video_ref, download_dir)
            emit_progress(
                options.progress,
                "download.complete",
                video_id=video_ref.video_id,
                size_bytes=audio_file.stat().st_size,
            )
            transcript = self.audio.transcribe(audio_file, audio_options)

            if options.output_dir is not None:
                base_name = slugify(video_ref.title or video_ref.video_id)
                transcript_dir = Path(options.output_dir) / base_name
                transcript.save(transcript_dir)
                emit_progress(
                    options.progress,
                    "export.complete",
                    video_id=video_ref.video_id,
                    path=str((transcript_dir / "transcript.txt").resolve()),
                )
                if not options.keep_audio:
                    shutil.rmtree(download_dir, ignore_errors=True)

        return VideoTranscriptResult(
            video=video_ref,
            status="transcribed",
            transcript=transcript,
        )

    @classmethod
    def from_openai(cls, *, api_key: str, base_url: str | None = None) -> YTTranscriber:
        from ai_providers import OpenAIOptions, create_ai
        from audio_transcribe import AudioTranscriber
        from audio_transcribe.backends import AIProviderTranscriptionBackend

        ai = create_ai(openai=OpenAIOptions(api_key=api_key, base_url=base_url))
        audio = AudioTranscriber(backend=AIProviderTranscriptionBackend(ai))
        return cls(audio=audio)

    def _transcribe_resolved_many(
        self,
        resolved: ResolvedSource,
        options: YTTranscribeOptions,
    ) -> MultiTranscriptResult:
        if resolved.source_type in {"playlist", "channel"}:
            emit_progress(
                options.progress,
                "listing.start",
                source_type=resolved.source_type,
            )
            videos = self.youtube.list_videos(
                resolved.source_url or "",
                options.listing,
            )
            emit_progress(
                options.progress,
                "listing.complete",
                source_type=resolved.source_type,
                videos=len(videos),
            )
        else:
            videos = resolved.videos

        results: list[VideoTranscriptResult] = []
        for video in videos:
            try:
                results.append(self.transcribe_video(video, options))
            except Exception as exc:  # noqa: BLE001 - batch should capture failure by option
                if not options.continue_on_error:
                    raise
                emit_progress(
                    options.progress,
                    "video.failed",
                    video_id=video.video_id,
                    error=str(exc),
                )
                results.append(
                    VideoTranscriptResult(
                        video=video,
                        status="failed",
                        error=str(exc),
                    )
                )

        return MultiTranscriptResult(
            source_type=resolved.source_type,  # type: ignore[arg-type]
            source_url=resolved.source_url,
            title=resolved.title,
            videos=tuple(results),
        )


def _audio_options_for(
    video: YouTubeVideoRef,
    options: YTTranscribeOptions,
) -> AudioTranscribeOptions:
    return AudioTranscribeOptions(
        language=options.language,
        prompt=options.prompt,
        model=options.model,
        source_id=f"youtube:{video.video_id}",
        skip_existing=options.skip_existing,
        force=options.force,
        keep_chunks=options.keep_audio,
        concurrency=options.concurrency,
        cache_dir=options.cache_dir,
        output_dir=options.output_dir,
        chunking=options.chunking,
        progress=options.progress,
    )


def _load_cached_audio_result(
    audio: AudioTranscriber,
    options: AudioTranscribeOptions,
) -> object | None:
    if not options.skip_existing or options.force or not options.source_id:
        return None
    cache = audio.cache if options.cache_dir is None else TranscriptCache(options.cache_dir)
    key = cache.key_for(
        source_id=options.source_id,
        backend=audio.backend.name,
        model=options.model or audio.backend.model,
        options=options,
    )
    return cache.load_result(key)
