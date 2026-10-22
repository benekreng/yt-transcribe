# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Minimal cross-platform clipboard access without third-party dependencies."""

from __future__ import annotations

import shutil
import subprocess
import sys


class ClipboardError(RuntimeError):
    """Raised when the system clipboard cannot be read or written."""


def read_clipboard() -> str:
    """Return the current clipboard text, or raise ClipboardError."""
    for command in _read_commands():
        if shutil.which(command[0]) is None:
            continue
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ClipboardError(f"Could not read clipboard via {command[0]}: {exc}") from exc
        return completed.stdout
    raise ClipboardError(_no_tool_message("read"))


def write_clipboard(text: str) -> None:
    """Replace the clipboard contents with ``text``, or raise ClipboardError."""
    for command in _write_commands():
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text,
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ClipboardError(f"Could not write clipboard via {command[0]}: {exc}") from exc
        return
    raise ClipboardError(_no_tool_message("write"))


def _read_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbpaste"]]
    if sys.platform.startswith("win"):
        return [["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"]]
    return [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]


def _write_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if sys.platform.startswith("win"):
        return [["clip"]]
    return [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard", "-i"],
        ["xsel", "--clipboard", "--input"],
    ]


def _no_tool_message(action: str) -> str:
    names = ", ".join(command[0] for command in (_read_commands() if action == "read" else _write_commands()))
    return f"No clipboard tool available to {action} the clipboard (looked for: {names})."
