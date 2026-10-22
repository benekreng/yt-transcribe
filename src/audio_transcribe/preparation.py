# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio probing and source preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from audio_transcribe.errors import AudioPreparationError, FFmpegNotFoundError
from audio_transcribe.models import PreparedAudio


def ensure_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FFmpegNotFoundError(f"{binary} not found on PATH.")


def hash_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def probe_audio(path: str | Path, *, ffprobe_path: str = "ffprobe") -> PreparedAudio:
    ensure_binary(ffprobe_path)
    audio_path = Path(path)
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "json",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AudioPreparationError(exc.stderr.strip() or "ffprobe failed") from exc

    data = json.loads(completed.stdout)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0)
    return PreparedAudio(
        source_id=hash_file(audio_path),
        file_path=str(audio_path),
        format=fmt.get("format_name"),
        duration_seconds=duration,
        size_bytes=audio_path.stat().st_size,
    )
