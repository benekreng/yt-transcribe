# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Static provider metadata.

Credentials do not belong in this module. Callers inject secrets when building
the provider facade.
"""

from __future__ import annotations

OPENAI_CONFIG = {
    "display_name": "OpenAI",
    "base_url": "https://api.openai.com/v1",
    "chat": {
        "model": "gpt-4o",
        "default_temperature": 0,
        "supports_streaming": True,
        "supports_vision": True,
    },
    "transcribe": {
        "model": "gpt-transcribe",
        "max_file_bytes": 25 * 1024 * 1024,
        "supported_formats": [
            "flac",
            "m4a",
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "oga",
            "ogg",
            "wav",
            "webm",
        ],
        "default_response_format": "json",
    },
    "embed": {
        "model": "text-embedding-3-small",
        "max_batch_size": 2048,
        "output_dimensions": 1536,
    },
    "image": {
        "model": "dall-e-3",
        "sizes": ["1024x1024", "1792x1024", "1024x1792"],
        "quality": "standard",
    },
}

PROVIDER_CONFIG = {
    "openai": OPENAI_CONFIG,
}
