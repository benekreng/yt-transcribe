# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Load CLI settings from environment files without changing the process environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values


class EnvironmentFileError(ValueError):
    """An environment file could not be read."""


def load_environment(
    *,
    env_file: Path | None = None,
    enabled: bool = True,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Merge user and project files, then let the process environment win.

    An explicit file replaces automatic discovery. Values are read literally,
    without variable interpolation or shell execution.
    """
    process = dict(os.environ if environ is None else environ)
    if not enabled:
        return process

    if env_file is not None:
        files = [env_file.expanduser()]
    else:
        config_home = Path(process.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        files = [
            config_home / "yt-transcribe" / ".env",
            (cwd if cwd is not None else Path.cwd()) / ".env",
        ]

    settings: dict[str, str] = {}
    for path in files:
        try:
            with path.open(encoding="utf-8-sig") as stream:
                values = dotenv_values(stream=stream, interpolate=False)
        except FileNotFoundError:
            if env_file is None:
                continue
            raise EnvironmentFileError(f"Environment file not found: {path}") from None
        except (OSError, UnicodeError):
            # Do not include file contents or decoder errors in CLI responses.
            raise EnvironmentFileError(f"Could not read environment file: {path}") from None
        settings.update({key: value for key, value in values.items() if value is not None})

    settings.update(process)
    return settings
