# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Image generation client for OpenAI-compatible image endpoints."""

from __future__ import annotations

from ai_providers.clients.http import OpenAIHTTPClient
from ai_providers.config.providers import OPENAI_CONFIG
from ai_providers.models import GeneratedImage, ImageOptions, ImageResult


class ImageClient:
    def __init__(self, http: OpenAIHTTPClient) -> None:
        self._http = http

    def image(self, prompt: str, options: ImageOptions | None = None) -> ImageResult:
        options = options or ImageOptions()
        model = options.model or str(OPENAI_CONFIG["image"]["model"])
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "n": options.count,
            "size": options.size or OPENAI_CONFIG["image"]["sizes"][0],
        }
        if model == "dall-e-3":
            payload["quality"] = options.quality or OPENAI_CONFIG["image"]["quality"]
            if options.style:
                payload["style"] = options.style

        data = self._http.post_json("/images/generations", payload)
        return ImageResult(
            images=tuple(
                GeneratedImage(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
                for item in data.get("data", [])
            )
        )
