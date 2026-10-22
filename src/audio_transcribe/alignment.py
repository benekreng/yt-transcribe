# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Give a transcript text timestamps by matching it against timed words.

Some transcription models write clean text without timestamps, others return
words with timestamps. Matching the two word sequences gives every word of the
clean text a time: matched words take the time of their counterpart, the rest
are interpolated between their matched neighbours.

Matching uses difflib, so keep inputs to a few thousand words (align chunk by
chunk for long recordings).
"""

from __future__ import annotations

import bisect
import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from audio_transcribe.models import TranscriptSegment

TokenSource = Literal["matched", "interpolated", "filled"]

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’»)\]]*$")
# Abbreviations whose full stop doesn't end a sentence ("Mr. Musk", "z.B. so").
_ABBREVIATIONS = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "e.g.", "i.e.", "z.b.", "bzw.", "nr.", "ca.", "u.a."}
)


@dataclass(frozen=True)
class TimedWord:
    """A word with a known time, e.g. from Whisper's word timestamps."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignedToken:
    """One whitespace-separated token of the aligned text."""

    text: str
    start: float
    end: float
    # matched: time taken from a matching timed word.
    # interpolated: no counterpart; time estimated from matched neighbours.
    # filled: the text had nothing here, so the timed word itself was inserted.
    source: TokenSource


@dataclass(frozen=True)
class Alignment:
    tokens: tuple[AlignedToken, ...]
    # Share of the text's words that matched a timed word.
    match_ratio: float
    # Time ranges where the text was missing speech and timed words were inserted.
    filled_gaps: tuple[tuple[float, float], ...]


def normalize_token(token: str) -> str:
    """Lowercase and strip punctuation so "Claude," matches "claude"."""

    return _NON_WORD.sub("", token.casefold())


def align_text_to_words(
    text: str,
    words: Sequence[TimedWord],
    *,
    fill_gaps_longer_than: float | None = None,
) -> Alignment:
    """Assign a time to every token of `text` using `words` as the clock.

    With `fill_gaps_longer_than`, stretches of at least that many seconds where
    `words` has speech but `text` has (almost) nothing are filled with the timed
    words themselves. This repairs text sources that silently drop passages.
    """

    tokens = text.split()
    normalized = [normalize_token(token) for token in tokens]
    timed = [word for word in words if normalize_token(word.text)]
    timed_normalized = [normalize_token(word.text) for word in timed]

    # Tokens that are pure punctuation (e.g. "—") take part in the output but not in matching.
    comparable = [index for index, value in enumerate(normalized) if value]
    matcher = difflib.SequenceMatcher(
        None,
        [normalized[index] for index in comparable],
        timed_normalized,
        autojunk=False,
    )

    times: list[tuple[float, float] | None] = [None] * len(tokens)
    dropped: set[int] = set()
    inserted: dict[int, list[TimedWord]] = {}
    gaps: list[tuple[float, float]] = []
    matched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                word = timed[j1 + offset]
                times[comparable[i1 + offset]] = (word.start, word.end)
            matched += i2 - i1
            continue
        if fill_gaps_longer_than is None or j2 == j1:
            continue
        span = timed[j1:j2]
        if span[-1].end - span[0].start < fill_gaps_longer_than:
            continue
        if (j2 - j1) <= 2 * (i2 - i1) + 3:
            continue  # both sides have comparable amounts of text: a disagreement, not a gap
        position = comparable[i1] if i1 < len(comparable) else len(tokens)
        dropped.update(comparable[i1:i2])
        inserted.setdefault(position, []).extend(span)
        gaps.append((span[0].start, span[-1].end))

    sequence: list[list] = []
    for index in range(len(tokens) + 1):
        for word in inserted.get(index, ()):
            sequence.append([word.text.strip(), word.start, word.end, "filled"])
        if index < len(tokens) and index not in dropped:
            known = times[index]
            if known is None:
                sequence.append([tokens[index], None, None, "interpolated"])
            else:
                sequence.append([tokens[index], known[0], known[1], "matched"])

    _interpolate(sequence, timed)
    return Alignment(
        tokens=tuple(AlignedToken(text, start, end, source) for text, start, end, source in sequence),
        match_ratio=matched / len(comparable) if comparable else 0.0,
        filled_gaps=tuple(gaps),
    )


def split_sentences(
    tokens: Sequence[AlignedToken],
    *,
    max_words: int = 80,
) -> tuple[TranscriptSegment, ...]:
    """Group aligned tokens into sentences using the text's own punctuation.

    Sentences longer than `max_words` (e.g. unpunctuated filled gaps) are split
    so downstream consumers always get bounded units.
    """

    sentences: list[TranscriptSegment] = []
    current: list[AlignedToken] = []

    def flush() -> None:
        if current:
            sentences.append(
                TranscriptSegment(
                    text=" ".join(token.text for token in current),
                    start=current[0].start,
                    end=max(current[-1].end, current[0].start),
                )
            )
            current.clear()

    for token in tokens:
        current.append(token)
        ends = _SENTENCE_END.search(token.text) and token.text.casefold() not in _ABBREVIATIONS
        if ends or len(current) >= max_words:
            flush()
    flush()
    return tuple(sentences)


def _interpolate(sequence: list[list], timed: Sequence[TimedWord]) -> None:
    known = [index for index, item in enumerate(sequence) if item[1] is not None]
    if not known:
        start = timed[0].start if timed else 0.0
        end = timed[-1].end if timed else 0.0
        step = (end - start) / max(len(sequence), 1)
        for index, item in enumerate(sequence):
            item[1] = item[2] = start + step * index
        return

    for index, item in enumerate(sequence):
        if item[1] is not None:
            continue
        position = bisect.bisect_left(known, index)
        previous = known[position - 1] if position > 0 else None
        following = known[position] if position < len(known) else None
        if previous is None:
            time = sequence[following][1]
        elif following is None:
            time = sequence[previous][2]
        else:
            left, right = sequence[previous][2], sequence[following][1]
            time = left + (right - left) * (index - previous) / (following - previous)
        item[1] = item[2] = time
