# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Transcription backend implementations."""

from audio_transcribe.backends.ai_provider import AIProviderTranscriptionBackend
from audio_transcribe.backends.base import TranscriptionBackend

__all__ = ["AIProviderTranscriptionBackend", "TranscriptionBackend"]
