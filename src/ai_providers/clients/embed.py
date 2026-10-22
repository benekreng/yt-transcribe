# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Embedding client for OpenAI-compatible embeddings."""

from __future__ import annotations

from ai_providers.clients.http import OpenAIHTTPClient
from ai_providers.config.providers import OPENAI_CONFIG
from ai_providers.models import EmbedOptions, EmbedResult


class EmbedClient:
    def __init__(self, http: OpenAIHTTPClient) -> None:
        self._http = http

    def embed(self, texts: list[str] | tuple[str, ...], options: EmbedOptions | None = None) -> EmbedResult:
        options = options or EmbedOptions()
        payload: dict[str, object] = {
            "model": options.model or OPENAI_CONFIG["embed"]["model"],
            "input": list(texts),
            "encoding_format": "float",
        }
        if options.dimensions is not None:
            payload["dimensions"] = options.dimensions

        data = self._http.post_json("/embeddings", payload)
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in items]
