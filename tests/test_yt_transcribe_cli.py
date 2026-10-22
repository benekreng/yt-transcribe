# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
from pathlib import Path

import pytest

from audio_transcribe import ProgressEvent, TranscriptResult
from yt_transcribe import (
    MultiTranscriptResult,
    VideoTranscriptResult,
    YouTubeVideoRef,
)
import yt_transcribe_cli.cli as cli_module


@pytest.fixture(autouse=True)
def isolate_environment_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "REMEM_TRANSCRIPTS_DIR"]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cli():
    return cli_module


def _video_result(*, status="transcribed"):
    transcript = TranscriptResult(
        text="hello from the video",
        segments=tuple(),
        language="en",
        duration_seconds=12,
        chunks=tuple(),
        source_id="youtube:abc123abc12",
        backend="fake",
        model="fake-model",
    )
    return VideoTranscriptResult(
        video=YouTubeVideoRef(
            video_id="abc123abc12",
            url="https://www.youtube.com/watch?v=abc123abc12",
            title="Example Video",
        ),
        status=status,
        transcript=transcript,
    )


class FakeTranscriber:
    def __init__(self, result, *, emit_progress=False):
        self.result = result
        self.emit_progress = emit_progress
        self.calls = []

    def transcribe(self, source, options):
        self.calls.append(("transcribe", source, options))
        if self.emit_progress and options.progress:
            options.progress(
                ProgressEvent(
                    name="metadata.complete",
                    details={"video_id": "abc123abc12", "title": "Example Video"},
                )
            )
        return self.result

    def multi_transcribe(self, source, options):
        self.calls.append(("multi_transcribe", source, options))
        return self.result

    def transcribe_many(self, sources, options):
        self.calls.append(("transcribe_many", sources, options))
        return self.result


@pytest.mark.parametrize("location", ["user", "project", "explicit"])
@pytest.mark.parametrize("key_name", ["OPENAI_API_KEY", "CUSTOM_TRANSCRIBE_KEY"])
def test_cli_reads_key_endpoint_and_output_from_env_file(
    cli, monkeypatch, tmp_path, capsys, location, key_name,
):
    path = {
        "user": tmp_path / "config" / "yt-transcribe" / ".env",
        "project": tmp_path / ".env",
        "explicit": tmp_path / "credentials.env",
    }[location]
    path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "saved output"
    path.write_text(
        f'{key_name}=saved-key\nOPENAI_BASE_URL=https://file.example/v1\n'
        f'REMEM_TRANSCRIPTS_DIR="{output_dir}"\n'
    )
    credentials = {}
    fake = FakeTranscriber(_video_result())

    def create_transcriber(**kwargs):
        credentials.update(kwargs)
        return fake

    monkeypatch.setattr(cli.YTTranscriber, "from_openai", create_transcriber)
    arguments = ["abc123abc12", "--json", "--api-key-env", key_name]
    if location == "explicit":
        arguments.extend(["--env-file", str(path)])

    assert cli.main(arguments) == cli.EXIT_OK
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert credentials == {"api_key": "saved-key", "base_url": "https://file.example/v1"}
    assert fake.calls[0][2].output_dir == output_dir
    assert payload["output_dir"] == str(output_dir)
    assert "saved-key" not in captured.out + captured.err


def test_cli_options_and_process_environment_override_file_settings(cli, monkeypatch, tmp_path, capsys):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://file.example/v1\nREMEM_TRANSCRIPTS_DIR=./file-output\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process.example/v1")
    monkeypatch.setenv("REMEM_TRANSCRIPTS_DIR", "./process-output")
    credentials = {}
    fake = FakeTranscriber(_video_result())

    def create_transcriber(**kwargs):
        credentials.update(kwargs)
        return fake

    monkeypatch.setattr(cli.YTTranscriber, "from_openai", create_transcriber)
    assert cli.main(["abc123abc12", "--json", "--base-url", "https://cli.example/v1", "-o", "./cli-output"]) == cli.EXIT_OK
    assert credentials == {"api_key": "process-key", "base_url": "https://cli.example/v1"}
    assert fake.calls[0][2].output_dir == tmp_path / "cli-output"
    capsys.readouterr()


def test_cli_no_env_file_disables_saved_credentials(cli, tmp_path, capsys):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=saved-key\n")

    assert cli.main(["abc123abc12", "--no-env-file", "--json"]) == cli.EXIT_USAGE_ERROR
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["type"] == "configuration_error"
    assert "saved-key" not in captured.out + captured.err


@pytest.mark.parametrize("kind", ["missing", "invalid-utf8"])
def test_cli_env_file_errors_keep_json_contract(cli, tmp_path, capsys, kind):
    path = tmp_path / "credentials.env"
    if kind == "invalid-utf8":
        path.write_bytes(b"OPENAI_API_KEY=sensitive-value\xff\n")

    assert cli.main(["abc123abc12", "--env-file", str(path), "--json"]) == cli.EXIT_USAGE_ERROR
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["type"] == "configuration_error"
    assert "sensitive-value" not in captured.out + captured.err


def test_json_error_when_api_key_is_missing(cli, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = cli.main(["https://youtu.be/abc123abc12", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert payload["ok"] is False
    assert payload["error"]["type"] == "configuration_error"
    assert "OPENAI_API_KEY" in payload["error"]["message"]


def test_json_video_response_is_stable_and_uses_absolute_artifact_paths(
    cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    fake = FakeTranscriber(_video_result())
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        cli.YTTranscriber,
        "from_openai",
        lambda **kwargs: fake,
    )

    exit_code = cli.main(
        [
            "abc123abc12",
            "--json",
            "--output-dir",
            str(tmp_path / "out"),
            "--language",
            "en",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert payload["ok"] is True
    assert payload["kind"] == "video"
    assert payload["video"]["video_id"] == "abc123abc12"
    assert payload["transcript"]["text"] == "hello from the video"
    assert payload["artifacts"]["text"] == str(
        (tmp_path / "out" / "example-video" / "transcript.txt").resolve()
    )
    assert fake.calls[0][2].language == "en"


def test_text_format_prints_only_transcript(cli, monkeypatch, capsys, tmp_path):
    fake = FakeTranscriber(_video_result(status="cached"))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        cli.YTTranscriber,
        "from_openai",
        lambda **kwargs: fake,
    )

    exit_code = cli.main(
        [
            "abc123abc12",
            "--format",
            "text",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_OK
    assert captured.out == "hello from the video\n"
    assert captured.err == ""


def test_text_is_default_and_verbose_events_go_only_to_stderr(
    cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    fake = FakeTranscriber(_video_result(), emit_progress=True)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        cli.YTTranscriber,
        "from_openai",
        lambda **kwargs: fake,
    )

    exit_code = cli.main(
        [
            "abc123abc12",
            "--verbose",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_OK
    assert captured.out == "hello from the video\n"
    assert "cli.start" in captured.err
    assert "metadata.complete" in captured.err
    assert 'title=\"Example Video\"' in captured.err
    assert "cli.complete" in captured.err


def test_collection_partial_failure_has_distinct_exit_code(
    cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    success = _video_result()
    failed = VideoTranscriptResult(
        video=YouTubeVideoRef(
            video_id="def456def45",
            url="https://www.youtube.com/watch?v=def456def45",
            title="Failed Video",
        ),
        status="failed",
        error="download failed",
    )
    result = MultiTranscriptResult(
        source_type="manual",
        source_url=None,
        title=None,
        videos=(success, failed),
    )
    fake = FakeTranscriber(result)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        cli.YTTranscriber,
        "from_openai",
        lambda **kwargs: fake,
    )

    exit_code = cli.main(
        [
            "abc123abc12",
            "def456def45",
            "--json",
            "--output-dir",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PARTIAL_FAILURE
    assert payload["ok"] is False
    assert payload["counts"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "combined.txt").read_text() == "hello from the video"


def test_operational_failure_is_clean_json(cli, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    def fail(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(cli.YTTranscriber, "from_openai", fail)

    exit_code = cli.main(["abc123abc12", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OPERATIONAL_ERROR
    assert payload == {
        "ok": False,
        "error": {
            "type": "RuntimeError",
            "message": "network unavailable",
        },
    }


class FakeClipboard:
    def __init__(self, content=""):
        self.content = content
        self.written = []

    def read(self):
        return self.content

    def write(self, text):
        self.written.append(text)


def _install_clipboard(monkeypatch, cli, content=""):
    clipboard = FakeClipboard(content)
    monkeypatch.setattr(cli, "read_clipboard", clipboard.read)
    monkeypatch.setattr(cli, "write_clipboard", clipboard.write)
    return clipboard


def test_no_source_reads_clipboard_and_copies_transcript_back(
    cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    fake = FakeTranscriber(_video_result())
    clipboard = _install_clipboard(
        monkeypatch, cli, "  https://youtu.be/abc123abc12\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(cli.YTTranscriber, "from_openai", lambda **kwargs: fake)

    exit_code = cli.main(["--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_OK
    assert fake.calls == [("transcribe", "https://youtu.be/abc123abc12", fake.calls[0][2])]
    assert captured.out == "hello from the video\n"
    assert "source (from clipboard): https://youtu.be/abc123abc12" in captured.err
    assert "copied transcript to clipboard" in captured.err
    assert clipboard.written == ["hello from the video"]


def test_explicit_source_does_not_touch_clipboard_unless_copy_requested(
    cli,
    monkeypatch,
    capsys,
    tmp_path,
):
    fake = FakeTranscriber(_video_result())
    clipboard = _install_clipboard(monkeypatch, cli, "https://youtu.be/zzzzzzzzzzz")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(cli.YTTranscriber, "from_openai", lambda **kwargs: fake)

    assert cli.main(["abc123abc12", "--output-dir", str(tmp_path)]) == cli.EXIT_OK
    assert fake.calls[-1][1] == "abc123abc12"
    assert clipboard.written == []

    assert cli.main(["abc123abc12", "--copy", "--output-dir", str(tmp_path)]) == cli.EXIT_OK
    assert clipboard.written == ["hello from the video"]
    assert capsys.readouterr().out == "hello from the video\nhello from the video\n"


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("", "clipboard is empty"),
        ("just some notes I copied", "not a YouTube source"),
        ("https://youtu.be/abc123abc12\nhttps://youtu.be/def456def45", "single URL"),
    ],
)
def test_no_source_with_unusable_clipboard_is_a_usage_error(
    cli,
    monkeypatch,
    capsys,
    content,
    fragment,
):
    _install_clipboard(monkeypatch, cli, content)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    exit_code = cli.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert payload["error"]["type"] == "configuration_error"
    assert fragment in payload["error"]["message"]


def test_clipboard_unavailable_reports_error_without_traceback(cli, monkeypatch, capsys):
    def unavailable():
        raise cli.ClipboardError("No clipboard tool available")

    monkeypatch.setattr(cli, "read_clipboard", unavailable)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE_ERROR
    assert "No clipboard tool available" in captured.err
    assert "hint:" in captured.err


def test_clipboard_write_failure_is_a_warning_not_an_error(cli, monkeypatch, capsys, tmp_path):
    fake = FakeTranscriber(_video_result())
    monkeypatch.setattr(cli, "read_clipboard", lambda: "abc123abc12")

    def broken(text):
        raise cli.ClipboardError("pbcopy exploded")

    monkeypatch.setattr(cli, "write_clipboard", broken)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(cli.YTTranscriber, "from_openai", lambda **kwargs: fake)

    exit_code = cli.main(["--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_OK
    assert captured.out == "hello from the video\n"
    assert "warning: pbcopy exploded" in captured.err
