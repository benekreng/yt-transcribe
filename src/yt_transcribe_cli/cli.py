#!/usr/bin/env python3
# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Human- and agent-friendly CLI for the Remem YouTube transcriber."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Sequence

from audio_transcribe import ChunkingOptions, ProgressEvent
from yt_transcribe import (
    ListVideosOptions,
    MultiTranscriptResult,
    VideoTranscriptResult,
    YTTranscribeOptions,
    YTTranscriber,
    resolve_source,
)
from yt_transcribe.cache import slugify
from yt_transcribe.errors import InvalidSourceError
from yt_transcribe_cli.clipboard import ClipboardError, read_clipboard, write_clipboard

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "REMEM_TRANSCRIPTS_DIR",
        Path.home() / ".remem" / "transcripts",
    )
)

EXIT_OK = 0
EXIT_OPERATIONAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_PARTIAL_FAILURE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-transcribe",
        description="Transcribe YouTube videos, playlists, or channels.",
        epilog=(
            "The OpenAI key is read from OPENAI_API_KEY by default. "
            "Use --format json for a stable machine-readable response. "
            "With no source, the URL is read from the clipboard and the "
            "finished transcript is copied back to it."
        ),
    )
    parser.add_argument(
        "source",
        nargs="*",
        help=(
            "One or more YouTube URLs or video IDs. "
            "Omit to read a single source from the clipboard."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help=(
            "Copy the finished transcript to the clipboard "
            "(implied when the source comes from the clipboard)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: <output-dir>/.cache).",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json", "text", "path"),
        default="text",
        help="Stdout format (default: text). 'path' prints the absolute transcript path.",
    )
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="format",
        help="Alias for --format json.",
    )
    parser.add_argument("--language", help="Language hint such as en or de.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Transcription prompt.")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the transcription prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--model",
        help=(
            "OpenAI transcription model (default: gpt-transcribe). "
            "gpt-4o-transcribe stops after 2,000 output tokens, so pair it "
            "with --chunk-duration 240 or less."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        metavar="NAME",
        help="Environment variable containing the API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL (or set OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completed and partial cached results.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not reuse a completed cached transcript.",
    )
    parser.add_argument("--keep-audio", action="store_true", help="Keep downloaded audio.")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=3,
        help="Concurrent chunk transcription requests (default: 3).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=_positive_float,
        default=600,
        metavar="SECONDS",
        help="Maximum chunk duration (default: 600).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=_positive_float,
        default=24,
        metavar="MB",
        help="Maximum chunk size (default: 24).",
    )
    parser.add_argument(
        "--overlap",
        type=_nonnegative_float,
        default=2,
        metavar="SECONDS",
        help="Chunk overlap (default: 2).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Treat an ambiguous watch URL containing list= as a collection.",
    )
    parser.add_argument(
        "--max-items",
        type=_positive_int,
        help="Maximum videos to process from a playlist or channel.",
    )
    parser.add_argument(
        "--include-shorts",
        action="store_true",
        help="Include Shorts in playlist/channel sources.",
    )
    parser.add_argument(
        "--include-livestreams",
        action="store_true",
        help="Include livestreams in playlist/channel sources.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop a collection when its first video fails.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Write timestamped progress events to stderr.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback for operational failures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.source:
        try:
            args.source = [_source_from_clipboard()]
        except ClipboardError as exc:
            return _configuration_error(
                args,
                str(exc),
                hint="Copy a YouTube URL or pass it as an argument.",
            )
        args.copy = True

    if len(args.source) > 1 and args.multi:
        return _configuration_error(
            args,
            "--multi cannot be combined with multiple explicit sources.",
        )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        return _configuration_error(
            args,
            f"{args.api_key_env} is not set.",
            hint=(
                f"Export the key in your shell or use --api-key-env NAME. "
                "The key itself is never accepted as a CLI argument."
            ),
        )

    try:
        prompt = (
            args.prompt_file.read_text(encoding="utf-8")
            if args.prompt_file is not None
            else args.prompt
        )
    except OSError as exc:
        return _configuration_error(args, f"Could not read prompt file: {exc}")

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / ".cache"
    )
    options = YTTranscribeOptions(
        language=args.language,
        prompt=prompt,
        model=args.model,
        skip_existing=not args.no_cache,
        force=args.force,
        continue_on_error=not args.fail_fast,
        concurrency=args.concurrency,
        cache_dir=cache_dir,
        output_dir=output_dir,
        keep_audio=args.keep_audio,
        chunking=ChunkingOptions(
            max_duration_seconds=args.chunk_duration,
            max_size_mb=args.chunk_size_mb,
            overlap_seconds=args.overlap,
        ),
        listing=ListVideosOptions(
            max_items=args.max_items,
            include_shorts=args.include_shorts,
            include_livestreams=args.include_livestreams,
        ),
        progress=EventLog() if args.verbose else None,
    )

    try:
        _progress(options.progress, "cli.start", sources=len(args.source))
        transcriber = YTTranscriber.from_openai(
            api_key=api_key,
            base_url=args.base_url,
        )
        if len(args.source) > 1:
            result = transcriber.transcribe_many(args.source, options)
        elif args.multi:
            result = transcriber.multi_transcribe(args.source[0], options)
        else:
            result = transcriber.transcribe(args.source[0], options)
    except Exception as exc:  # noqa: BLE001 - CLI converts library failures to a stable contract
        _progress(
            options.progress,
            "cli.failed",
            error_type=type(exc).__name__,
            error=str(exc) or type(exc).__name__,
        )
        if args.debug:
            traceback.print_exc()
        return _operational_error(args, exc)

    response = _success_response(args.source, result, output_dir)
    _progress(
        options.progress,
        "cli.complete",
        status=(
            "partial"
            if isinstance(result, MultiTranscriptResult) and result.failures()
            else "ok"
        ),
        path=_primary_artifact(response, result),
    )
    _emit_success(args.format, response, result)

    if args.copy:
        _copy_result_to_clipboard(result)

    if isinstance(result, MultiTranscriptResult) and result.failures():
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


def _source_from_clipboard() -> str:
    """Read a single YouTube source from the clipboard, validating it up front."""
    source = read_clipboard().strip()
    if not source:
        raise ClipboardError("No source given and the clipboard is empty.")
    if "\n" in source or len(source) > 2048:
        raise ClipboardError("No source given and the clipboard does not contain a single URL.")
    try:
        resolve_source(source)
    except InvalidSourceError:
        raise ClipboardError(
            f"No source given and the clipboard is not a YouTube source: {_preview(source)}"
        ) from None
    print(f"source (from clipboard): {source}", file=sys.stderr, flush=True)
    return source


def _copy_result_to_clipboard(
    result: VideoTranscriptResult | MultiTranscriptResult,
) -> None:
    if isinstance(result, VideoTranscriptResult):
        text = result.transcript.text if result.transcript is not None else ""
    else:
        text = result.combined_text()
    if not text:
        print("warning: nothing to copy to the clipboard.", file=sys.stderr)
        return
    try:
        write_clipboard(text)
    except ClipboardError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return
    print(f"copied transcript to clipboard ({len(text)} characters).", file=sys.stderr)


def _preview(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _success_response(
    sources: Sequence[str],
    result: VideoTranscriptResult | MultiTranscriptResult,
    output_dir: Path,
) -> dict[str, Any]:
    if isinstance(result, VideoTranscriptResult):
        return {
            "ok": True,
            "kind": "video",
            "sources": list(sources),
            "output_dir": str(output_dir),
            "status": result.status,
            "video": result.video.to_dict(),
            "transcript": result.transcript.to_dict() if result.transcript else None,
            "artifacts": _video_artifacts(result, output_dir),
        }

    result_path = output_dir / "result.json"
    result.save(output_dir)
    failures = len(result.failures())
    return {
        "ok": failures == 0,
        "kind": "collection",
        "sources": list(sources),
        "output_dir": str(output_dir),
        "counts": {
            "total": len(result.videos),
            "succeeded": len(result.successes()),
            "failed": failures,
        },
        "result": result.to_dict(),
        "artifacts": {
            "result_json": str(result_path),
            "combined_text": str(output_dir / "combined.txt"),
            "videos": [
                _video_artifacts(item, output_dir)
                for item in result.videos
                if item.transcript is not None
            ],
        },
    }


def _video_artifacts(result: VideoTranscriptResult, output_dir: Path) -> dict[str, str]:
    base_name = slugify(result.video.title or result.video.video_id)
    directory = output_dir / base_name
    return {
        "directory": str(directory),
        "text": str(directory / "transcript.txt"),
        "json": str(directory / "transcript.json"),
        "srt": str(directory / "transcript.srt"),
        "vtt": str(directory / "transcript.vtt"),
    }


def _primary_artifact(
    response: dict[str, Any],
    result: VideoTranscriptResult | MultiTranscriptResult,
) -> str:
    if isinstance(result, VideoTranscriptResult):
        return response["artifacts"]["text"]
    return response["artifacts"]["combined_text"]


def _emit_success(
    output_format: str,
    response: dict[str, Any],
    result: VideoTranscriptResult | MultiTranscriptResult,
) -> None:
    if output_format == "json":
        print(json.dumps(response, ensure_ascii=False))
        return

    if output_format == "text":
        if isinstance(result, VideoTranscriptResult):
            assert result.transcript is not None
            print(result.transcript.text)
        else:
            print(result.combined_text())
        return

    artifacts = response["artifacts"]
    if output_format == "path":
        if isinstance(result, VideoTranscriptResult):
            print(artifacts["text"])
        else:
            print(artifacts["combined_text"])
        return

    if isinstance(result, VideoTranscriptResult):
        print(f"status: {result.status}")
        print(f"title: {result.video.title or result.video.video_id}")
        print(f"video_id: {result.video.video_id}")
        print(f"transcript: {artifacts['text']}")
        print(f"json: {artifacts['json']}")
    else:
        counts = response["counts"]
        print(
            f"status: {'partial' if counts['failed'] else 'ok'} "
            f"({counts['succeeded']}/{counts['total']} succeeded)"
        )
        print(f"transcript: {artifacts['combined_text']}")
        print(f"json: {artifacts['result_json']}")


def _configuration_error(
    args: argparse.Namespace,
    message: str,
    *,
    hint: str | None = None,
) -> int:
    payload = {
        "ok": False,
        "error": {
            "type": "configuration_error",
            "message": message,
            "hint": hint,
        },
    }
    _emit_error(args.format, payload)
    return EXIT_USAGE_ERROR


def _operational_error(args: argparse.Namespace, exc: Exception) -> int:
    payload = {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
        },
    }
    _emit_error(args.format, payload)
    return EXIT_OPERATIONAL_ERROR


def _emit_error(output_format: str, payload: dict[str, Any]) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False))
        return
    error = payload["error"]
    print(f"error: {error['message']}", file=sys.stderr)
    if error.get("hint"):
        print(f"hint: {error['hint']}", file=sys.stderr)


class EventLog:
    """Line-oriented progress renderer safe for redirected stdout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self, event: ProgressEvent) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        details = " ".join(
            f"{key}={_event_value(value)}"
            for key, value in event.details.items()
            if value is not None
        )
        line = f"{timestamp}  {event.name:<24}"
        if details:
            line += f" {details}"
        with self._lock:
            print(line.rstrip(), file=sys.stderr, flush=True)


def _event_value(value: Any) -> str:
    if isinstance(value, str):
        if value and all(character.isalnum() or character in "._:/@+-" for character in value):
            return value
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _progress(callback, name: str, **details: Any) -> None:
    if callback is not None:
        callback(ProgressEvent(name=name, details=details))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
