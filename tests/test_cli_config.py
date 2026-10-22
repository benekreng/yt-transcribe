# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

import pytest

from yt_transcribe_cli.config import EnvironmentFileError, load_environment


def test_user_then_project_then_process_precedence_without_mutating_environment(tmp_path):
    config_home = tmp_path / "config"
    user = config_home / "yt-transcribe" / ".env"
    user.parent.mkdir(parents=True)
    user.write_text("OPENAI_API_KEY=user-key\nOPENAI_BASE_URL=https://user.example/v1\n")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=project-key\nREMEM_TRANSCRIPTS_DIR=./out\n")
    process = {"XDG_CONFIG_HOME": str(config_home), "OPENAI_API_KEY": "process-key"}
    original = process.copy()

    settings = load_environment(environ=process, cwd=tmp_path)

    assert settings["OPENAI_API_KEY"] == "process-key"
    assert settings["OPENAI_BASE_URL"] == "https://user.example/v1"
    assert settings["REMEM_TRANSCRIPTS_DIR"] == "./out"
    assert process == original
    assert load_environment(environ={"XDG_CONFIG_HOME": str(config_home)}, cwd=tmp_path)["OPENAI_API_KEY"] == "project-key"


def test_default_user_config_location(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    user = tmp_path / ".config" / "yt-transcribe" / ".env"
    user.parent.mkdir(parents=True)
    user.write_text("OPENAI_API_KEY=saved-key\n")

    assert load_environment(environ={}, cwd=tmp_path)["OPENAI_API_KEY"] == "saved-key"


def test_explicit_file_replaces_automatic_files_but_not_process_values(tmp_path):
    user = tmp_path / "config" / "yt-transcribe" / ".env"
    user.parent.mkdir(parents=True)
    user.write_text("USER_ONLY=ignored\n")
    (tmp_path / ".env").write_text("PROJECT_ONLY=ignored\n")
    explicit = tmp_path / "credentials.env"
    explicit.write_text("OPENAI_API_KEY=explicit-key\nOPENAI_BASE_URL=https://file.example/v1\n")

    settings = load_environment(
        env_file=explicit,
        environ={"XDG_CONFIG_HOME": str(tmp_path / "config"), "OPENAI_BASE_URL": "https://process.example/v1"},
        cwd=tmp_path,
    )

    assert settings["OPENAI_API_KEY"] == "explicit-key"
    assert settings["OPENAI_BASE_URL"] == "https://process.example/v1"
    assert "USER_ONLY" not in settings
    assert "PROJECT_ONLY" not in settings


def test_disabled_files_use_only_process_environment(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=file-key\n")
    assert load_environment(enabled=False, environ={"OPENAI_API_KEY": "process-key"}, cwd=tmp_path) == {"OPENAI_API_KEY": "process-key"}


def test_missing_automatic_files_are_optional_and_parent_directories_are_not_searched(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=parent-key\n")
    child = tmp_path / "child"
    child.mkdir()
    process = {"XDG_CONFIG_HOME": str(tmp_path / "missing-config")}

    assert load_environment(environ=process, cwd=child) == process


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(EnvironmentFileError, match="Environment file not found"):
        load_environment(env_file=tmp_path / "missing.env", environ={})


@pytest.mark.parametrize("kind", ["directory", "invalid-utf8"])
def test_unreadable_files_report_paths_without_exposing_contents(tmp_path, kind):
    path = tmp_path / "credentials.env"
    if kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"OPENAI_API_KEY=sensitive-value\xff\n")

    with pytest.raises(EnvironmentFileError, match="Could not read environment file") as error:
        load_environment(env_file=path, environ={})
    assert "sensitive-value" not in str(error.value)


def test_dotenv_quotes_comments_bom_and_literal_values(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text(
        "\ufeff# comment\nexport OPENAI_API_KEY='${SECRET}$(touch should-not-exist)'\n"
        'CUSTOM_KEY="value with spaces" # comment\nBARE_KEY\nEMPTY_KEY=\n',
        encoding="utf-8",
    )

    settings = load_environment(env_file=path, environ={"SECRET": "must-not-expand"})

    assert settings["OPENAI_API_KEY"] == "${SECRET}$(touch should-not-exist)"
    assert settings["CUSTOM_KEY"] == "value with spaces"
    assert settings["EMPTY_KEY"] == ""
    assert "BARE_KEY" not in settings
    assert not (tmp_path / "should-not-exist").exists()


def test_empty_process_value_does_not_fall_back_to_file_secret(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text("OPENAI_API_KEY=file-key\n")

    assert load_environment(env_file=path, environ={"OPENAI_API_KEY": ""})["OPENAI_API_KEY"] == ""
