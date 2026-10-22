# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from audio_transcribe import ChunkingOptions, plan_chunk_windows


def test_plan_chunk_windows_adds_overlap_and_preserves_nominal_ranges():
    windows = plan_chunk_windows(
        1300,
        ChunkingOptions(max_duration_seconds=600, overlap_seconds=2),
    )

    assert [(w.start, w.end, w.nominal_start, w.nominal_end) for w in windows] == [
        (0.0, 602.0, 0.0, 600.0),
        (598.0, 1202.0, 600.0, 1200.0),
        (1198.0, 1300, 1200.0, 1300),
    ]
