# aloud

> Turn any web article or PDF into a private, narrated **read-along** — on your Mac, fully offline.

![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-black)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/runs-100%25%20offline-success)

`aloud` is a local text-to-speech reader for Apple Silicon Macs. Point it at a
URL, a PDF, or your clipboard and it reads the content aloud in a natural neural
voice — **Kokoro v1.0** via **MLX-Audio**, Metal-accelerated on the GPU — with a
**karaoke-style highlight** following the words in your terminal and
**YouTube-style playback controls**. No cloud, no subscriptions, nothing leaves
your machine.

![aloud demo](assets/demo.svg)

```sh
aloud https://example.com/some-long-article      # fetch, de-clutter, and read it aloud
aloud paper.pdf --pages 12-18                     # read just a section
aloud --clipboard --speed 1.2                     # read whatever you copied
```

---

## Why aloud?

- **Private & offline.** Everything runs on-device. After the one-time model
  download (~330 MB) you can pull the network cable and it still works. No API
  keys, no accounts, no per-character billing.
- **Reads the article, not the chrome.** Web pages are stripped of nav, ads, and
  boilerplate down to the actual prose before a word is spoken.
- **A real read-along.** The current paragraph is shown word-wrapped in your
  terminal with the spoken word highlighted; headings and lists keep their shape;
  code blocks (which sound awful aloud) are skipped with a short notice.
- **Controls you already know.** `space`/`k` pause, `j`/`l` seek ±10s, arrows ±5s,
  `[`/`]` jump paragraphs — just like a video player.
- **Starts fast, stays responsive.** Audio streams chunk-by-chunk, so playback
  begins a second or two after the model loads instead of after the whole
  article. A single `Ctrl-C` stops everything cleanly — no orphaned processes.
- **Lean & reproducible.** One readable script, a `uv`-managed environment, and
  deliberately **no PyTorch** (see [below](#kept-deliberately-lean)).

---

## What it does

- `aloud <url>` — fetch a page, strip nav/ads/boilerplate to the article text, and read it
- `aloud <file.pdf>` — extract and read a PDF; `--pages 3-7` for a page range
- `aloud --text "…"` — read a literal string
- `aloud --clipboard` — read whatever is on the clipboard (`pbpaste`)
- `--voice <name>` — pick a voice (default `af_bella`); see `aloud --list-voices`
- `--speed <n>` — speech speed (default `1.0`)
- `--save out.wav` — save audio to a file instead of playing

---

## Playback controls (interactive)

When you play to your speakers (i.e. not `--save`, and you're in a real
terminal), the now-playing paragraph is shown with a moving word highlight, above
a live control bar:

```
▶  Playing   ¶ 7/42   ████████░░░░░░░░░░░░░░░░   3:48 / 12:10
```

It shows play/pause state, which paragraph you're on (`¶ 7/42`), a progress bar,
and elapsed / total time. A trailing `+` on the time means the rest is still
being synthesized in the background.

Keys follow YouTube's conventions:

| Key | Action |
|-----|--------|
| `space` or `k` | play / pause |
| `j` / `l` | seek back / forward 10 s |
| `←` / `→` | seek back / forward 5 s |
| `[` / `]` | previous / next paragraph |
| `0` | restart from the beginning |
| `q` or `Ctrl-C` | quit |

Backward seeks are always instant; forward seeks go as far as has been
synthesized so far (synthesis quickly runs ahead of playback). Everything stays
in memory — playback never writes to disk.

> The word highlight is positioned by elapsed time within each paragraph (Kokoro
> doesn't emit real per-word timestamps), so it tracks closely but isn't
> sample-exact.

> Piping/redirecting output (no TTY) falls back to plain streaming with
> `chunk N/M` progress and no key controls.

---

## Install

Requires [`uv`](https://docs.astral.sh/uv/). Everything else is handled by `uv`.

Run `aloud` directly from GitHub without installing it:

```sh
uvx --from git+https://github.com/wasim/aloud aloud https://example.com/article
```

Or install the `aloud` command permanently:

```sh
uv tool install git+https://github.com/wasim/aloud
aloud https://example.com/article
```

To work from a clone instead, create the environment and command with:

```sh
git clone https://github.com/wasim/aloud.git ~/aloud
cd ~/aloud
uv sync
uv run aloud https://example.com/article
```

The whole environment is reproducible from `pyproject.toml` + `uv.lock` with a
single `uv sync`.

### First run

The first time you read anything, it downloads the Kokoro model (~330 MB) from
Hugging Face and a small spaCy English model (pinned as a dependency). Both are
cached; after that, it works **offline**.

> **Why Python 3.13 and not 3.14?** Kokoro's English G2P (`misaki` → `spacy`)
> has no 3.14 wheels yet (`spacy` ships `cp313` only). 3.13 is the newest version
> where the whole stack installs from prebuilt wheels. `requires-python` is
> pinned to `>=3.13,<3.14` so `uv sync` won't drift onto 3.14 and break.

> <a name="kept-deliberately-lean"></a>**Kept deliberately lean.** The obvious
> install, `misaki[en]`, pulls in **PyTorch (~2 GB)** and a transformer stack
> that Kokoro's default G2P never touches. Instead this project installs only the
> pieces it actually uses (`misaki`, `spacy`, `num2words`, plus `espeakng-loader`
> + `phonemizer-fork` for the espeak-ng fallback that pronounces
> out-of-dictionary proper nouns), and pins the small spaCy English model
> directly. No Torch, much smaller env.

---

## Examples

```sh
aloud https://claude.com/blog/how-anthropic-enables-self-service-data-analytics-with-claude
aloud paper.pdf --pages 3-7
aloud --text "The quick brown fox." --voice af_heart
aloud --clipboard --speed 1.2
aloud longread.pdf --save longread.wav     # save instead of play
aloud --list-voices
```

---

## Model & voices

**Model:** `mlx-community/Kokoro-82M-bf16` — Kokoro v1.0, a top-ranked
open-weight TTS model with excellent quality-per-compute and reliable
pronunciation on clean English text.

**Default voice:** `af_bella` — chosen for long-form listening comfort and
stable pronunciation over hour-plus articles without artifacts.

**Recommended alternative:** `af_heart` — extremely clean; the model's default
voice. Worth A/B testing on your own text:

```sh
aloud --text "your sample paragraph" --voice af_bella
aloud --text "your sample paragraph" --voice af_heart
```

`aloud --list-voices` shows everything available (US/UK male & female, plus
other-language voices). Voice prefixes: `af`/`am` = US female/male,
`bf`/`bm` = UK female/male. The first letter sets the language (a=US English,
b=UK English), so picking a `bf_`/`bm_` voice reads in British English
automatically.

### Pronunciation note

Kokoro is tuned for clean, well-punctuated English (American/British) in a
neutral/informational tone — ideal for blog posts, docs, and PDFs. Pronunciation
of unusual proper nouns or acronyms can occasionally slip (it falls back to
espeak-ng phonemes for out-of-dictionary words). That's a known limitation of
the model, not a bug in this tool.

---

## Stopping it / checking nothing is left running

A **single Ctrl-C** (or `q`) immediately stops synthesis *and* playback and exits
cleanly — no orphaned `python` processes, no audio left playing, and your
terminal restored to normal.

```sh
pgrep -fl "mlx_audio|aloud.py"     # should be empty after you quit
pkill -f aloud.py                  # force-kill, if ever needed
```

### How it stays kill-safe

- Audio plays through an in-process [`sounddevice`](https://python-sounddevice.readthedocs.io/)
  stream (PortAudio) — there is no external `afplay`/player subprocess to orphan.
- Ctrl-C sets a single `quit` flag that every loop checks, so shutdown is
  deterministic (more reliable than racing a `KeyboardInterrupt` out of a
  blocking wait).
- The synth, playback, and keyboard threads are all daemons watching that flag,
  so none can outlive the main process or keep the GPU/audio busy after you quit.
- The terminal is put into cbreak mode for single-key controls and always
  restored on exit (even on crash), via a `finally` block.

---

## Files

- `aloud.py` — the whole tool (one readable script)
- `pyproject.toml` / `uv.lock` — reproducible environment
- `~/.local/bin/aloud` — the command installed by `uv tool install`

---

## License

[MIT](LICENSE).

This project uses but does not bundle: the Kokoro v1.0 model (Apache-2.0),
MLX-Audio, misaki, spaCy, trafilatura, and pypdf — each under its own license.
