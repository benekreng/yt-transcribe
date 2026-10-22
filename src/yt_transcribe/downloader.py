# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""YouTube audio download adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from yt_transcribe.errors import DownloadError
from yt_transcribe.models import YouTubeVideoRef


class AudioDownloader(Protocol):
    def download_audio(self, video: YouTubeVideoRef, output_dir: str | Path) -> Path:
        ...


class YtDlpAudioDownloader:
    def __init__(self, *, yt_dlp_path: str = "yt-dlp") -> None:
        self.yt_dlp_path = yt_dlp_path

    def download_audio(self, video: YouTubeVideoRef, output_dir: str | Path) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        template = path / f"{video.video_id}.%(ext)s"
        try:
            subprocess.run(
                [
                    self.yt_dlp_path,
                    "--no-playlist",
                    "-f",
                    "bestaudio/best",
                    "-o",
                    str(template),
                    video.url,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(exc.stderr.strip() or "yt-dlp audio download failed") from exc

        matches = sorted(path.glob(f"{video.video_id}.*"))
        if not matches:
            raise DownloadError(f"yt-dlp did not produce an audio file for {video.video_id}.")
        return matches[0]
