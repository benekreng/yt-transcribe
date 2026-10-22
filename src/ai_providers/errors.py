# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Provider-layer errors."""

from __future__ import annotations

from collections.abc import Mapping


class ProviderError(RuntimeError):
    """Base error for provider calls."""


class ProviderHTTPError(ProviderError):
    """HTTP error returned by a provider."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        # Response headers, e.g. retry-after on 429 responses.
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}


class ProviderConfigurationError(ProviderError):
    """Invalid provider configuration."""
