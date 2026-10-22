# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json

from ai_providers import TranscribeOptions
from ai_providers.clients.http import _build_multipart_body
from ai_providers.clients.transcribe import TranscriptionClient
from ai_providers.errors import ProviderHTTPError


class FakeHTTP:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.fields = None

    def post_multipart(self, path, *, fields, file):
        self.fields = fields
        return "application/json", json.dumps(self.response).encode("utf-8")


def test_verbose_json_with_word_timestamps_is_requested_and_parsed():
    http = FakeHTTP(
        {
            "text": "Hello there.",
            "language": "english",
            "duration": 1.5,
            "segments": [
                {
                    "text": " Hello there.",
                    "start": 0.0,
                    "end": 1.5,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01,
                }
            ],
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.6},
                {"word": "there", "start": 0.7, "end": 1.4},
            ],
        }
    )

    result = TranscriptionClient(http).transcribe(
        b"audio",
        TranscribeOptions(
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=("word", "segment"),
        ),
    )

    assert http.fields["timestamp_granularities[]"] == ["word", "segment"]
    assert [(w.text, w.start, w.end) for w in result.words] == [("Hello", 0.0, 0.6), ("there", 0.7, 1.4)]
    assert result.segments[0].compression_ratio == 1.1
    assert result.segments[0].no_speech_prob == 0.01
    assert result.language == "english"


def test_plain_json_request_has_no_granularity_field_and_reads_languages_list():
    http = FakeHTTP({"text": "Hallo.", "languages": [{"code": "de"}]})

    result = TranscriptionClient(http).transcribe(b"audio", TranscribeOptions(model="gpt-transcribe"))

    assert "timestamp_granularities[]" not in http.fields
    assert result.language == "de"
    assert result.words == ()


def test_default_model_is_gpt_transcribe_and_language_is_sent_as_languages_list():
    http = FakeHTTP({"text": "Hallo.", "languages": [{"code": "de"}]})

    TranscriptionClient(http).transcribe(b"audio", TranscribeOptions(language="de"))

    assert http.fields["model"] == "gpt-transcribe"
    assert http.fields["languages[]"] == ["de"]
    assert "language" not in http.fields


def test_older_models_keep_the_singular_language_field():
    http = FakeHTTP({"text": "Hallo."})

    TranscriptionClient(http).transcribe(
        b"audio",
        TranscribeOptions(model="gpt-4o-transcribe", language="de"),
    )

    assert http.fields["language"] == "de"
    assert "languages[]" not in http.fields


def test_multipart_body_repeats_sequence_fields():
    body = _build_multipart_body(
        boundary="b",
        fields={"model": "m", "timestamp_granularities[]": ["word", "segment"]},
        file_field="file",
        filename="a.mp3",
        file_bytes=b"x",
        content_type="audio/mpeg",
    ).decode()

    assert body.count('name="timestamp_granularities[]"') == 2
    assert "word\r\n" in body and "segment\r\n" in body


def test_http_error_keeps_lowercased_headers():
    error = ProviderHTTPError("rate limited", status=429, body="", headers={"Retry-After": "7"})

    assert error.headers == {"retry-after": "7"}
    assert ProviderHTTPError("x", status=500, body="").headers == {}


def test_requests_send_a_named_user_agent():
    from ai_providers.clients.http import OpenAIHTTPClient

    headers = OpenAIHTTPClient(api_key="k", base_url="https://api.groq.com/openai/v1")._headers()

    # Cloudflare-fronted providers answer 403 (error 1010) to Python's default User-Agent.
    assert headers["User-Agent"].startswith("remem-ai-providers/")
