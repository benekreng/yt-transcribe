<div align="center">

# yt-transcribe

**YouTube transcripts for your terminal, clipboard, and scripts.**

[![CI](https://github.com/benekreng/yt-transcribe/actions/workflows/ci.yml/badge.svg)](https://github.com/benekreng/yt-transcribe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![MPL-2.0 License](https://img.shields.io/badge/license-MPL--2.0-green)](LICENSE)

Videos, playlists, and channels → text, JSON, SRT, and VTT.

</div>

```bash
yt-transcribe "https://youtu.be/VIDEO_ID"
yt-transcribe "https://youtu.be/VIDEO_ID" --copy
yt-transcribe                         # URL from clipboard → transcript back to clipboard
```

Downloads audio with `yt-dlp`, splits it with FFmpeg, and sends the chunks to
an OpenAI-compatible transcription API. It transcribes the audio rather than
retrieving YouTube captions.

## Why use it?

- **Clipboard workflow.** Copy a YouTube URL, run `yt-transcribe`, and paste the transcript.
- **Shell friendly.** Transcript text goes to stdout; optional progress goes to stderr.
- **Ready for scripts and agents.** JSON responses, absolute artifact paths, and predictable exit codes.
- **Long recordings and collections.** Overlapping chunks, concurrent API calls, and cached chunk results for resuming interrupted work.
- **Four exports.** Every newly transcribed video is saved as text, JSON, SRT, and VTT.

## Install

You need Python **3.11+**, [uv](https://docs.astral.sh/uv/), and the
`yt-dlp`, `ffmpeg`, and `ffprobe` commands on your `PATH`.
FFmpeg installations normally include `ffprobe`.

On macOS with Homebrew:

```bash
brew install uv ffmpeg yt-dlp
uv tool install git+https://github.com/benekreng/yt-transcribe.git
uv tool ensurepath
```

On Debian or Ubuntu, after installing uv:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
uv tool install yt-dlp
uv tool install git+https://github.com/benekreng/yt-transcribe.git
uv tool ensurepath
```

Open a new terminal if `uv tool ensurepath` updates your shell configuration.
The Python package bundles its internal libraries and has no third-party
Python runtime dependencies; `yt-dlp` and FFmpeg are separate command-line tools.

Set your API key in the shell, then transcribe a video:

```bash
export OPENAI_API_KEY="your-api-key"
yt-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --verbose
```

The API receives the audio chunks and may charge for transcription.
Downloads, processing, and saved artifacts live on your machine.
The CLI reads credentials from environment variables, not command-line arguments,
and does not load `.env` files automatically.

## Everyday use

### Clipboard in, transcript out

Copy a video URL or ID, then run:

```bash
yt-transcribe
```

The transcript replaces your clipboard contents, prints to the terminal, and
is saved to disk. With an explicit source, add `--copy` for the same clipboard output.

Clipboard access uses `pbpaste`/`pbcopy` on macOS, `wl-clipboard`, `xclip`, or
`xsel` on Linux, and PowerShell/`clip` on Windows. Linux clipboard mode needs
one of those tools installed and a working desktop session.

### Pipe, save, or get a path

```bash
yt-transcribe "https://youtu.be/VIDEO_ID" > transcript.txt
yt-transcribe "https://youtu.be/VIDEO_ID" --verbose | less
yt-transcribe "https://youtu.be/VIDEO_ID" --format path
```

`--format path` prints the absolute path to the saved text file. For a
collection, it prints the path to `combined.txt`.
`--format human` prints a short status and artifact summary.

### Playlists, channels, and batches

```bash
yt-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-items 10
yt-transcribe "https://www.youtube.com/@CHANNEL" --max-items 10
yt-transcribe VIDEO_ID_1 VIDEO_ID_2 --json
```

A watch URL containing both `v=` and `list=` selects the video by default;
add `--multi` to process its playlist. Collection listing excludes Shorts and
livestreams by default; use `--include-shorts` and `--include-livestreams`
to include them. Collections continue after a failed video unless you pass
`--fail-fast`.

### Language, prompts, and endpoints

```bash
yt-transcribe VIDEO_ID --language de
yt-transcribe VIDEO_ID --prompt "Technical vocabulary: FFT, wavetable, oscillator."
yt-transcribe VIDEO_ID --prompt-file vocabulary.txt
yt-transcribe VIDEO_ID --model YOUR_MODEL --base-url https://your-provider.example/v1
```

The bundled configuration selects `gpt-transcribe` by default. Use `--model`
to select a model available at your endpoint, and adjust chunk length to its
input and output limits. `OPENAI_BASE_URL` also sets the endpoint;
`--base-url` takes precedence. Use `--api-key-env NAME` to read credentials
from a different environment variable.

## Output and caching

The default output directory is `~/.remem/transcripts`, preserved from the
original Remem tool. Set `REMEM_TRANSCRIPTS_DIR` or pass `-o` / `--output-dir`
to choose another location:

```bash
yt-transcribe VIDEO_ID --output-dir ./transcripts
```

Each new video transcript gets a directory named from its title, containing:

```text
transcript.txt
transcript.json
transcript.srt
transcript.vtt
```

Collections also write `combined.txt` and `result.json` at the output root.
Stdout text is the plain transcript; the saved `.txt` includes timestamps
when segments are present. Subtitle timing follows the API's segments;
if the API returns only text, timing falls back to whole chunks.

Cache files live under `<output-dir>/.cache` unless `--cache-dir` is set.
Completed transcripts and individual chunks can be reused on subsequent runs.
`--no-cache` bypasses completed-result reuse; chunk reuse still applies.
Use `--force` to bypass both. `--keep-audio` retains the downloaded audio
after a successful transcription.

Chunk defaults are 600 seconds, 24 MB, 2 seconds of overlap, and 3 concurrent
transcription requests. Tune them with `--chunk-duration`, `--chunk-size-mb`,
`--overlap`, and `--concurrency`:

```bash
yt-transcribe VIDEO_ID --chunk-duration 240 --concurrency 2
```

## JSON and automation

```bash
yt-transcribe VIDEO_ID --json
yt-transcribe VIDEO_ID --json | jq -r '.artifacts.text'
```

`--json` is an alias for `--format json`. A successful single-video response
includes `ok`, `kind`, `sources`, `output_dir`, `status`, `video`, `transcript`,
and `artifacts`. Collections include counts, per-video results, and combined
artifact paths. A collection with any failures returns `"ok": false` and exit code 3.

Configuration and operational failures also produce one JSON object, for example:

```json
{"ok": false, "error": {"type": "configuration_error", "message": "OPENAI_API_KEY is not set.", "hint": "..."}}
```

Exit codes: **0** success · **1** operational failure · **2** invalid arguments
or configuration · **3** partially failed collection or batch.
Argument parsing errors use the standard argparse message on stderr.
`--verbose` adds progress on stderr; `--debug` adds tracebacks for operational failures.

Run `yt-transcribe --help` for the full option reference.

## Python API

The same installation includes the underlying libraries:

```python
import os
from pathlib import Path
from yt_transcribe import YTTranscribeOptions, YTTranscriber

yt = YTTranscriber.from_openai(api_key=os.environ["OPENAI_API_KEY"])
result = yt.transcribe(
    "https://youtu.be/VIDEO_ID",
    YTTranscribeOptions(output_dir=Path("transcripts")),
)
print(result.transcript.text)
```

`YTTranscriber` also exposes `list_videos`, `transcribe_many`, and
`multi_transcribe`. Pass `progress=callback` in `YTTranscribeOptions` to
receive structured progress events.

## Development

```bash
git clone https://github.com/benekreng/yt-transcribe.git
cd yt-transcribe
uv sync --locked
uv run yt-transcribe --help
uv run pytest
uv build
```

Source lives in `src/`: `yt_transcribe_cli` handles the command and clipboard;
`yt_transcribe` handles YouTube orchestration; `audio_transcribe` handles audio,
chunking, caching, and exports; `ai_providers` supplies the shared provider
client. These are bundled in one distribution, with no Remem checkout or
private package registry required.

The test suite uses fake API and YouTube clients and generated audio, so it
needs no API key or network access. Install FFmpeg to run the audio integration
tests; otherwise those tests are skipped. CI runs on Python 3.11 and 3.14,
builds the package, and verifies installation of the resulting wheel.

Extracted from the Python CLI in `remem_modules`.
Copyright © 2026 Benedikt Krengel. Licensed under [MPL-2.0](LICENSE).
