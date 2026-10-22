# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Small stdlib HTTP helper for OpenAI-compatible JSON and multipart calls."""

from __future__ import annotations

import json
import mimetypes
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_providers.errors import ProviderHTTPError, ProviderError
from ai_providers.models import AudioFileInput

USER_AGENT = "remem-ai-providers/0.1"


class OpenAIHTTPClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        organization: str | None = None,
        project: str | None = None,
        timeout: float = 600,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        self.project = project
        self.timeout = timeout

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path),
            data=body,
            method="POST",
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
        )
        return self._read_json(request)

    def stream_json(self, path: str, payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path),
            data=body,
            method="POST",
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    yield json.loads(data)
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise ProviderHTTPError(
                f"Provider request failed ({exc.code}): {body_text}",
                status=exc.code,
                body=body_text,
                headers=dict(exc.headers.items()) if exc.headers else None,
            ) from exc
        except URLError as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc

    def post_multipart(
        self,
        path: str,
        *,
        fields: Mapping[str, str | Sequence[str]],
        file: AudioFileInput,
        file_field: str = "file",
        fallback_filename: str = "audio.mp3",
    ) -> tuple[str, bytes]:
        filename, file_bytes, content_type = _read_audio_file(file, fallback_filename)
        boundary = f"----remem-{uuid.uuid4().hex}"
        body = _build_multipart_body(
            boundary=boundary,
            fields=fields,
            file_field=file_field,
            filename=filename,
            file_bytes=file_bytes,
            content_type=content_type,
        )
        request = Request(
            self._url(path),
            data=body,
            method="POST",
            headers={
                **self._headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_type = response.headers.get_content_type()
                return response_type, response.read()
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise ProviderHTTPError(
                f"Provider request failed ({exc.code}): {body_text}",
                status=exc.code,
                body=body_text,
                headers=dict(exc.headers.items()) if exc.headers else None,
            ) from exc
        except URLError as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc

    def _read_json(self, request: Request) -> Mapping[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise ProviderHTTPError(
                f"Provider request failed ({exc.code}): {body_text}",
                status=exc.code,
                body=body_text,
                headers=dict(exc.headers.items()) if exc.headers else None,
            ) from exc
        except URLError as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        # A named User-Agent: Cloudflare-fronted providers (e.g. Groq) reject Python's default one.
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"


def _read_audio_file(file: AudioFileInput, fallback_filename: str) -> tuple[str, bytes, str]:
    if isinstance(file, bytes | bytearray):
        return fallback_filename, bytes(file), "audio/mpeg"

    if isinstance(file, str | Path):
        path = Path(file)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.name, path.read_bytes(), content_type

    name = Path(getattr(file, "name", fallback_filename)).name
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    data = file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return name, data, content_type


def _build_multipart_body(
    *,
    boundary: str,
    fields: Mapping[str, str | Sequence[str]],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        # A sequence value repeats the field, e.g. timestamp_granularities[]=word&...=segment.
        for item in [value] if isinstance(value, str) else value:
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    item.encode("utf-8"),
                    b"\r\n",
                ]
            )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks)
