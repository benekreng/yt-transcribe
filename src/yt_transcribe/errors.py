# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""YouTube transcription errors."""

from __future__ import annotations


class YTTranscribeError(RuntimeError):
    """Base error for YouTube transcription."""


class InvalidSourceError(YTTranscribeError):
    """The input source cannot be resolved."""


class YouTubeClientError(YTTranscribeError):
    """YouTube metadata/listing failed."""


class DownloadError(YTTranscribeError):
    """Audio download failed."""
