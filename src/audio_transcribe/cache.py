# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""File-system transcript cache."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_transcribe.models import ChunkTranscript, TranscriptResult
from audio_transcribe.options import AudioTranscribeOptions


SCHEMA_VERSION = 1


class TranscriptCache:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or ".cache/audio_transcribe")

    def key_for(
        self,
        *,
        source_id: str,
        backend: str,
        model: str,
        options: AudioTranscribeOptions,
    ) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_id": source_id,
            "backend": backend,
            "model": model,
            "language": options.language,
            "prompt": options.prompt,
            "chunking": _chunking_key(options),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def result_path(self, key: str) -> Path:
        return self.root / key / "final.json"

    def chunk_path(self, key: str, chunk_index: int) -> Path:
        return self.root / key / "chunks" / f"{chunk_index:04d}.json"

    def load_result(self, key: str) -> TranscriptResult | None:
        path = self.result_path(key)
        if not path.exists():
            return None
        return TranscriptResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_result(self, key: str, result: TranscriptResult) -> None:
        _atomic_write_json(self.result_path(key), result.to_dict())

    def load_chunk(self, key: str, chunk_index: int) -> ChunkTranscript | None:
        path = self.chunk_path(key, chunk_index)
        if not path.exists():
            return None
        return ChunkTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_chunk(self, key: str, chunk: ChunkTranscript) -> None:
        _atomic_write_json(self.chunk_path(key, chunk.chunk.index), asdict(chunk))


# Fields added after cache keys were first written. They are left out of the key
# while unset, so existing caches keep their keys.
_OPTIONAL_CHUNKING_FIELDS = ("bitrate",)


def _chunking_key(options: AudioTranscribeOptions) -> dict[str, Any]:
    return {
        name: value
        for name, value in asdict(options.chunking).items()
        if not (name in _OPTIONAL_CHUNKING_FIELDS and value is None)
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)
