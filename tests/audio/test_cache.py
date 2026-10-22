# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from audio_transcribe import AudioTranscribeOptions, ChunkingOptions, TranscriptCache


def test_cache_key_is_stable_and_includes_transcription_options(tmp_path):
    cache = TranscriptCache(tmp_path)
    options = AudioTranscribeOptions(
        language="en",
        prompt="Names are technical.",
        chunking=ChunkingOptions(max_duration_seconds=300),
    )

    key_a = cache.key_for(
        source_id="youtube:abc123",
        backend="openai",
        model="gpt-4o-transcribe",
        options=options,
    )
    key_b = cache.key_for(
        source_id="youtube:abc123",
        backend="openai",
        model="gpt-4o-transcribe",
        options=options,
    )
    key_c = cache.key_for(
        source_id="youtube:abc123",
        backend="openai",
        model="gpt-4o-transcribe",
        options=AudioTranscribeOptions(language="de"),
    )

    assert key_a == key_b
    assert key_a != key_c


def test_cache_keys_written_before_the_bitrate_option_are_unchanged(tmp_path):
    import hashlib
    import json
    from dataclasses import asdict

    options = AudioTranscribeOptions(language="en")
    legacy_chunking = {k: v for k, v in asdict(options.chunking).items() if k != "bitrate"}
    legacy_payload = {
        "schema_version": 1,
        "source_id": "youtube:abc123",
        "backend": "openai",
        "model": "gpt-4o-transcribe",
        "language": "en",
        "prompt": None,
        "chunking": legacy_chunking,
    }
    legacy_key = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    cache = TranscriptCache(tmp_path)
    key = cache.key_for(source_id="youtube:abc123", backend="openai", model="gpt-4o-transcribe", options=options)
    with_bitrate = cache.key_for(
        source_id="youtube:abc123",
        backend="openai",
        model="gpt-4o-transcribe",
        options=AudioTranscribeOptions(language="en", chunking=ChunkingOptions(bitrate="32k")),
    )

    assert key == legacy_key
    assert with_bitrate != legacy_key
