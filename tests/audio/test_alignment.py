# Copyright (c) 2026 Benedikt Krengel
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from audio_transcribe import TimedWord, align_text_to_words, split_sentences


def words(*items):
    return [TimedWord(text, start, start + 0.3) for text, start in items]


def test_matched_tokens_take_word_times_and_mismatches_are_interpolated():
    alignment = align_text_to_words(
        "and, key word here, and prompting Claude.",
        words(("and", 14.2), ("keyword", 14.5), ("here", 14.9), ("and", 15.1), ("prompting", 15.3), ("Claude", 15.8)),
    )

    by_text = {token.text: token for token in alignment.tokens}
    assert by_text["and,"].start == 14.2 and by_text["and,"].source == "matched"
    assert by_text["Claude."].start == 15.8
    assert by_text["key"].source == "interpolated"
    assert 14.5 <= by_text["key"].start <= by_text["word"].start <= 14.9
    assert round(alignment.match_ratio, 2) == round(5 / 7, 2)


def test_punctuation_and_case_do_not_prevent_matches():
    alignment = align_text_to_words("Hello, World!", words(("hello", 1.0), ("world", 2.0)))

    assert [token.source for token in alignment.tokens] == ["matched", "matched"]
    assert alignment.match_ratio == 1.0


def test_long_passage_missing_from_text_is_filled_from_timed_words():
    timed = words(("we", 0.0), ("start", 0.5)) + [
        TimedWord(f"w{i}", 1.0 + i, 1.5 + i) for i in range(30)
    ] + words(("the", 40.0), ("end", 40.5))

    alignment = align_text_to_words("We start. The end.", timed, fill_gaps_longer_than=15)

    filled = [token for token in alignment.tokens if token.source == "filled"]
    assert len(filled) == 30
    assert alignment.filled_gaps == ((1.0, 30.5),)
    assert [token.text for token in alignment.tokens][:3] == ["We", "start.", "w0"]


def test_short_disagreements_are_not_treated_as_gaps():
    alignment = align_text_to_words(
        "the path of the easiest life",
        words(("the", 0), ("path", 0.4), ("of", 0.8), ("the", 1.0), ("easy", 1.2), ("way", 1.4), ("of", 1.6), ("least", 1.8), ("resistance", 2.0)),
        fill_gaps_longer_than=15,
    )

    assert alignment.filled_gaps == ()
    assert [token.text for token in alignment.tokens] == "the path of the easiest life".split()


def test_text_without_any_match_is_spread_over_the_timed_range():
    alignment = align_text_to_words("ganz anderer Text", words(("other", 10.0), ("words", 12.0)))

    assert [token.source for token in alignment.tokens] == ["interpolated"] * 3
    assert alignment.tokens[0].start == 10.0
    assert alignment.match_ratio == 0.0


def test_split_sentences_uses_text_punctuation_and_bounds_length():
    alignment = align_text_to_words(
        "Is it bad? No. It is fine",
        words(("is", 1), ("it", 2), ("bad", 3), ("no", 4), ("it", 5), ("is", 6), ("fine", 7)),
    )

    sentences = split_sentences(alignment.tokens)
    assert [(s.text, s.start) for s in sentences] == [("Is it bad?", 1), ("No.", 4), ("It is fine", 5)]

    long = align_text_to_words(" ".join(["word"] * 10), words(*[("word", float(i)) for i in range(10)]))
    assert [len(s.text.split()) for s in split_sentences(long.tokens, max_words=4)] == [4, 4, 2]


def test_abbreviations_do_not_end_sentences():
    alignment = align_text_to_words(
        "What will they do, Mr. Musk? Dr. Smith agrees.",
        words(("what", 1), ("will", 2), ("they", 3), ("do", 4), ("mr", 5), ("musk", 6), ("dr", 7), ("smith", 8), ("agrees", 9)),
    )

    assert [s.text for s in split_sentences(alignment.tokens)] == ["What will they do, Mr. Musk?", "Dr. Smith agrees."]
